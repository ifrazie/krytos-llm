def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

def subtract_two_numbers(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

# Dictionary mapping function names to their implementations
available_functions = {
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
}
