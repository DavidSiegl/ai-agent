import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompts import system_prompt
from call_function import available_functions, call_function


def main():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    messages = [types.Content(
        role="user", parts=[types.Part(text=args.user_prompt)])]

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise RuntimeError("Unable to retrieve API KEY")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[available_functions], system_instruction=system_prompt),
    )

    prompt_tokens = response.usage_metadata.prompt_token_count
    response_tokens = response.usage_metadata.candidates_token_count
    if args.verbose == True:
        print(f"User prompt: {args.user_prompt}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Response tokens: {response_tokens}")
    if not response.function_calls:
        print("Response:")
        print(response.text)
        return

    function_results = []
    for function_call in response.function_calls:
        function_call_result = call_function(
            function_call, verbose=args.verbose)
        if not function_call_result.parts:
            raise ValueError(
                f"Function {function_call.name} returned empty parts.")
        first_part = function_call_result.parts[0]
        if first_part.function_response is None:
            raise ValueError(
                f"Function {function_call.name} result missing 'function_response'.")
        if first_part.function_response.response is None:
            raise ValueError(
                f"Function {function_call.name} returned None in 'response' field.")
        function_results.append(first_part)
        if args.verbose:
            print(f"-> {first_part.function_response.response}")


if __name__ == "__main__":
    main()
