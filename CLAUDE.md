# Mist Project Guidelines

## Build System
- Run `make` from the project root to build all examples and run tests
- Examples are in `examples/` subdirectories with their own Makefiles
- Tests are in `tests/` and run automatically as part of `make`

## Code Style

### C++ Standards
- C++20 with concepts, `std::same_as`, `std::type_identity`
- Prefer concepts over SFINAE for template constraints
- Use `auto` for return types when the type is clear from context
- Prefer `auto function() -> return_type` instead of `return_type function()`
- Use `auto var = type()` instead of `type var`

### Formatting
- 4-space indentation
- No indentation on empty lines
- Opening braces on same line
- No spaces inside parentheses or angle brackets
- Alphabetize `#include` directives

### Naming Conventions
- `snake_case` for functions, variables, namespaces
- `snake_case_t` suffix for type aliases and nested types (e.g., `config_t`, `state_t`)
- `PascalCase` for concepts (e.g., `Physics`, `HasFields`)
- `SCREAMING_CASE` for macros only

### Struct Design
- Provide both const and non-const `fields()` methods for serializable structs
- Use inline member initialization for defaults
- Group related fields together

### Namespaces
- Main namespace is `mist`
- Use nested namespaces for subsystems (e.g., `mist::driver`)
- ADL functions (`to_string`, `from_string`) go in the same namespace as their types

### Enums
- Use `enum class` for type safety
- Provide ADL `to_string(EnumType)` and `from_string(std::type_identity<EnumType>, const std::string&)`
- Place enums in appropriate namespace (e.g., `driver::scheduling_policy`)

### Error Handling
- Use `std::runtime_error` with descriptive messages
- Validate configuration early in `run()` functions

### Comments
- Minimal comments; code should be self-documenting
- Use section headers with `// ===` for major sections
- Use `// ---` for subsections within a namespace

## Git Conventions
- Commit frequently
- AI proposes a commit message before proceeding with a commit
- Commit messages start with message of 60 characters or fewer
- Bulleted content follows wrapped at column 72
