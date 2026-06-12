import nacl.utils
from nacl.public import PrivateKey

def generate_keypair():
    # Generate a secure random private key
    private_key = PrivateKey.generate()
    # Derive the public key
    public_key = private_key.public_key

    # Save Private Key (NEVER commit this)
    with open("developer_private.key", "wb") as f:
        f.write(bytes(private_key))

    # Save Public Key (Safe to put on server)
    with open("server_public.key", "wb") as f:
        f.write(bytes(public_key))

    print("Successfully generated server_public.key and developer_private.key")
    print("KEEP developer_private.key SAFE AND OFFLINE.")

if __name__ == "__main__":
    generate_keypair()
