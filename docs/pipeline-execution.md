# Pipeline Execution Algorithm

This document describes the generalized pipeline execution algorithm in `mist::parallel`. The algorithm coordinates multiple peers through a sequence of Exchange and Compute stages, allowing peers to proceed eagerly as soon as their dependencies are satisfied.

## Overview

A pipeline consists of a sequence of stages, each either:
- **Exchange**: peers request guard zone data from neighbors
- **Compute**: peers perform local computation

Peers progress through stages independently. A peer advances to the next stage as soon as its requirements are met, without waiting for all peers to complete the current stage.

## Data Structures

### Per-Peer State

```
peer_stage[peer_idx] : std::size_t
```
Which stage each peer is currently at (0 to num_stages-1, or num_stages if complete).

### Per-Stage Exchange State

```
guards_filled[stage_idx] : bitset<MaxPeers>
    Set when all of a peer's guard requests have been fulfilled.

requests_collected[stage_idx] : bitset<MaxPeers>
    Set when a peer's guard requests have been collected (happens once on arrival).

requesters_served[stage_idx * MaxPeers + provider_idx] : bitset<MaxPeers>
    For each provider at each stage, tracks which requesters have been served.

pending_requests[stage_idx * MaxPeers * R + peer_idx * R + r] : request_t
    The r-th guard request for peer_idx at stage_idx.
    Contains: buffer pointer, requested index space, fulfilled flag.

request_fulfilled[stage_idx * MaxPeers + peer_idx] : bitset<R>
    Tracks which of peer_idx's requests have been fulfilled.
```

### Per-Stage Compute State

```
compute_spawned[stage_idx] : bitset<MaxPeers>
    Set when a peer's compute task has been submitted to the scheduler.

compute_done[stage_idx * MaxPeers + peer_idx] : atomic<bool>
    Set by the compute task when complete.
```

## Algorithm

```
initialize:
    peer_stage[*] = 0  (all peers start at stage 0)
    all bitsets = empty
    completed_count = 0

while completed_count < num_peers:
    for each stage_idx in 0..num_stages:
        if stage is Exchange:
            process_exchange_stage(stage_idx)
        else:
            process_compute_stage(stage_idx)

    for each peer_idx in 0..num_peers:
        if can_advance(peer_idx):
            peer_stage[peer_idx]++
            if peer_stage[peer_idx] == num_stages:
                completed_count++

    yield()  // allow compute tasks to progress
```

### Process Exchange Stage

For each peer at this stage:

1. **Collect requests** (once per peer per stage):
   ```
   if not requests_collected[stage][peer]:
       Exchange::need(ctx, |buffer| {
           pending_requests[...] = {buffer, space(buffer)}
       })
       requests_collected[stage].set(peer)
   ```

2. **Try to fill requests**:
   ```
   for each unfulfilled request r:
       for each provider at this stage:
           if topo.owns(provider, request.space):
               Exchange::fill(provider_ctx, |src| {
                   topo.copy(request.buffer, src, request.space)
               })
               request_fulfilled[stage][peer].set(r)
               requesters_served[stage][provider].set(peer)
               break
   ```

3. **Check completion**:
   ```
   if all requests fulfilled:
       guards_filled[stage].set(peer)
   ```

### Process Compute Stage

For each peer at this stage:

1. **Spawn compute** (once per peer per stage):
   ```
   if not compute_spawned[stage].test(peer):
       compute_spawned[stage].set(peer)
       scheduler.spawn(|| {
           ctx = Compute::value(move(ctx))
           compute_done[stage][peer].store(true)
       })
   ```

### Advancement Conditions

**Exchange stage**: A peer can advance when:
1. All its guard requests are fulfilled (`guards_filled[stage].test(peer)`)
2. All potential requesters have been served

```
potential = topo.potential_requesters(peer)
served = requesters_served[stage][peer]
can_advance = guards_filled[stage].test(peer)
           && (potential & ~served).none()
```

**Compute stage**: A peer can advance when:
```
can_advance = compute_done[stage][peer].load()
```

## Topology Requirements

The topology must provide:

```cpp
// Copy data from src to dst for the requested region
void copy(buffer_t& dst, const buffer_t& src, space_t requested_space) const;

// Returns true if two spaces are neighbors (could exchange guards)
bool connected(space_t a, space_t b) const;
```

The `overlaps(space_a, space_b)` function from `core.hpp` is used to determine if a provider's space overlaps a request. The `connected()` method identifies potential requesters based on spatial adjacency.

For a 1D periodic domain, `connected(a, b)` returns true if `a` and `b` are adjacent (with wrapping).

## Why Peers Wait for Requesters

A peer cannot proceed to the next stage until all potential requesters have been served. This is necessary because:

1. The next stage might be Compute, which modifies the peer's data
2. Later-arriving requesters would then receive stale/corrupt data
3. This is a **consumer-driven** model - providers don't know who needs their data until requesters arrive

The alternative would be to snapshot/copy providable data when a peer wants to advance, but this adds memory overhead and the topology cannot know a priori what subset of data will be requested.

## Complexity

- **Space**: O(num_stages × num_peers × max_requests_per_peer) for request storage
- **Time per iteration**: O(num_stages × num_peers² × max_requests_per_peer) worst case for request matching

The algorithm uses fixed-size arrays and bitsets to minimize heap allocation. All state arrays are sized at compile time based on `MaxPeers` and `NumStages` template parameters.

## Example: 1D Advection with 4 Patches

```
Stage 0: Exchange (ghost zones)
Stage 1: Compute (flux calculation)

Initial: all peers at stage 0

Iteration 1:
  - All peers collect guard requests (need left and right neighbor data)
  - Peer 0 and Peer 2 happen to be processed first
  - They can provide data to each other but are waiting for Peers 1, 3

Iteration 2:
  - Peers 1, 3 arrive at stage 0
  - All requests can now be fulfilled
  - Peers check: guards filled? yes. all requesters served? yes.
  - All peers advance to stage 1

Iteration 3:
  - All peers spawn compute tasks
  - As each completes, it advances past stage 1
  - completed_count reaches 4, loop exits
```

With load imbalance (e.g., Peer 2 has more work), faster peers would wait at Exchange stage 0 only for their immediate neighbors, not for all peers globally.
