import requests

def test_profile_query():
    response = requests.get(
        "http://127.0.0.1:8000/ask",
        params={"question": "Tell me about Albion."},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()

    assert result["context_used"], "No profile context was retrieved"
    context = " ".join(result["context_used"]).lower()
    assert "albion" in context, "Retrieved context does not mention Albion"
    assert result["answer"].strip(), "The model returned an empty answer"

    print("✅ Profile query test passed")

if __name__ == "__main__":
    test_profile_query()
    print("All semantic tests passed!")
