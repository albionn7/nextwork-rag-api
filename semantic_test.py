import requests

def test_profile_query():
    response = requests.get(
        "http://127.0.0.1:8000/ask",
        params={"question": "What technologies does Albion have experience with?"},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()

    assert result["context_used"], "No profile context was retrieved"
    context = " ".join(result["context_used"]).lower()
    assert "typescript" in context, "Retrieved context does not mention TypeScript"
    assert "python" in context, "Retrieved context does not mention Python"
    assert result["answer"].strip(), "The model returned an empty answer"

    print("✅ Profile query test passed")

if __name__ == "__main__":
    test_profile_query()
    print("All semantic tests passed!")
