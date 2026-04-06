// AES-256-GCM encryption for sensitive data at rest.
// Edge-compatible — uses only crypto.subtle (no Node crypto).
// Requires CREDENTIALS_ENCRYPTION_KEY env var (64-char hex = 32 bytes).

const KEY_HEX = process.env.CREDENTIALS_ENCRYPTION_KEY ?? "";

async function getKey(): Promise<CryptoKey> {
  if (!KEY_HEX || KEY_HEX.length !== 64) {
    throw new Error(
      "CREDENTIALS_ENCRYPTION_KEY must be a 64-character hex string (32 bytes)"
    );
  }
  const raw = new Uint8Array(
    KEY_HEX.match(/.{2}/g)!.map((b) => parseInt(b, 16))
  );
  return crypto.subtle.importKey("raw", raw, "AES-GCM", false, [
    "encrypt",
    "decrypt",
  ]);
}

/**
 * Encrypt plaintext → base64 string (iv + ciphertext + tag).
 */
export async function encrypt(plaintext: string): Promise<string> {
  const key = await getKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encoded = new TextEncoder().encode(plaintext);

  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    encoded
  );

  // Combine: 12-byte IV + ciphertext (includes 16-byte auth tag)
  const combined = new Uint8Array(iv.length + ciphertext.byteLength);
  combined.set(iv);
  combined.set(new Uint8Array(ciphertext), iv.length);

  return btoa(String.fromCharCode(...combined));
}

/**
 * Decrypt base64 string → plaintext.
 */
export async function decrypt(cipherBase64: string): Promise<string> {
  const key = await getKey();
  const combined = Uint8Array.from(atob(cipherBase64), (c) => c.charCodeAt(0));

  const iv = combined.slice(0, 12);
  const ciphertext = combined.slice(12);

  const decrypted = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv },
    key,
    ciphertext
  );

  return new TextDecoder().decode(decrypted);
}
