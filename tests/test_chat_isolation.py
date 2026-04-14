from src.services.chat_service import scope_thread_id
import hashlib

def test_scope_thread_id_determinstic_hashing():
    """Verify that thread_id parameters are reliably and uniquely scoped using API Key entropy."""
    thread_id = "session-123"
    api_key_a = "tenant-a-secret-key"
    api_key_b = "tenant-b-secret-key"

    scoped_a = scope_thread_id(api_key_a, thread_id)
    scoped_b = scope_thread_id(api_key_b, thread_id)

    # They should never collide for different tenants
    assert scoped_a != scoped_b
    assert scoped_a.endswith(":" + thread_id)
    assert scoped_b.endswith(":" + thread_id)

    # Identical params yield identical scope maps (deterministic)
    assert scope_thread_id(api_key_a, thread_id) == scoped_a

    # Verify SHA256 was used for the prefix securely
    prefix_a = scoped_a.split(":")[0]
    expected_prefix = hashlib.sha256(api_key_a.encode()).hexdigest()[:16]
    assert prefix_a == expected_prefix
