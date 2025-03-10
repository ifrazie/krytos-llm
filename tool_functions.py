import requests
from datetime import datetime

def get_user_name_and_email_and_id(__user__: dict = {}) -> str:
    """
    Get the user name, Email and ID from the user object.
    """

    # Do not include :param for __user__ in the docstring as it should not be shown in the tool's specification
    # The session user object will be passed as a parameter when the function is called

    print(__user__)
    result = ""

    if "name" in __user__:
        result += f"User: {__user__['name']}"
    if "id" in __user__:
        result += f" (ID: {__user__['id']})"
    if "email" in __user__:
        result += f" (Email: {__user__['email']})"

    if result == "":
        result = "User: Unknown"

    return result


def get_current_time() -> str:
    """
    Get the current time in a more human-readable format.
    :return: The current time.
    """

    now = datetime.now()
    current_time = now.strftime("%I:%M:%S %p")  # Using 12-hour format with AM/PM
    current_date = now.strftime(
        "%A, %B %d, %Y"
    )  # Full weekday, month name, day, and year

    return f"Current Date and Time = {current_date}, {current_time}"


def calculator(equation: str) -> str:
    """
    Calculate the result of an equation.
    :param equation: The equation to calculate.
    """

    # Avoid using eval in production code
    # https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    try:
        result = eval(equation)
        return f"{equation} = {result}"
    except Exception as e:
        print(e)
        return "Invalid equation"


def get_current_weather(city: str) -> dict:
    """
    Get the current weather for a specified city.
    Args:
        city (str): The city to get weather information for
    Returns:
        dict: A formatted response containing the weather information
    """
    try:
        base_url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(base_url)
        data = response.json()

        weather_info = {
            "temperature": data['current_condition'][0]['temp_F'],
            "city": city,
            "response": f"The current temperature in {city} is {data['current_condition'][0]['temp_F']}°F"
        }

        return {
            "status": "success",
            "data": weather_info,
            "message": weather_info["response"]
        }
    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "message": f"Error getting weather for {city}: {str(e)}"
        }


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
    'get_current_weather': get_current_weather,
    'calculator': calculator,
    'get_current_time': get_current_time,
    'get_user_name_and_email_and_id': get_user_name_and_email_and_id
}
