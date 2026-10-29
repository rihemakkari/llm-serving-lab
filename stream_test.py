import time
from openai import OpenAI

# 1. Point the official OpenAI client at OUR server instead of OpenAI's.
#    The api_key is required by the library but ignored by llama.cpp (no key set).
client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="pas-de-cle")

question = "Explique en quatre phrases pourquoi on code les réponses ouvertes d'une enquête."

start = time.perf_counter()        # precise stopwatch
first_token_time = None
n_chunks = 0
answer = ""

# 2. stream=True: the server sends the answer piece by piece as it is generated,
#    instead of waiting for the whole text.
stream = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": question}],
    temperature=0,
    stream=True,
)

for chunk in stream:
    piece = chunk.choices[0].delta.content if chunk.choices else None
    if piece:
        if first_token_time is None:
            first_token_time = time.perf_counter()   # moment the 1st token arrives
        n_chunks += 1
        answer += piece
        print(piece, end="", flush=True)             # show it live

end = time.perf_counter()

# 3. The three numbers that matter for a serving service:
ttft = first_token_time - start                       # time to first token
gen_time = end - first_token_time                     # time spent generating
print("\n\n--- Mesures ---")
print(f"TTFT (temps avant le 1er token) : {ttft:.3f} s")
print(f"Morceaux reçus (≈ tokens)       : {n_chunks}")
print(f"Vitesse de génération           : {n_chunks / gen_time:.1f} tokens/s")
print(f"Durée totale                    : {end - start:.2f} s")