from together import Together


def main():
    """
    Simple smoke test for Together.ai chat completion.
    Make sure TOGETHER_API_KEY is set in your environment before running.
    """
    client = Together()  # auth defaults to os.environ.get("TOGETHER_API_KEY")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": "What are some fun things to do in New York?",
            }
        ],
    )

    # Print the first choice content
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

