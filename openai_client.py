from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://127.0.0.1:8000/v1",
)

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="reference",
    input="Hello, thank you for calling. How may I help you today?",
    response_format="wav",
) as response:
    response.stream_to_file("openai_output.wav")

print("Saved openai_output.wav")
