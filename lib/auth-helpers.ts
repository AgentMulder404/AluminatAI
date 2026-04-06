// Timing-safe string comparison for Edge runtime.
// Uses HMAC to avoid timing side-channels on secret comparison.

const HMAC_KEY_MATERIAL = new Uint8Array(32);
crypto.getRandomValues(HMAC_KEY_MATERIAL);

let _hmacKey: CryptoKey | null = null;

async function getHmacKey(): Promise<CryptoKey> {
  if (!_hmacKey) {
    _hmacKey = await crypto.subtle.importKey(
      "raw",
      HMAC_KEY_MATERIAL,
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"]
    );
  }
  return _hmacKey;
}

/**
 * Constant-time string comparison.
 * Returns true if a === b without leaking timing information.
 */
export async function constantTimeEqual(
  a: string,
  b: string
): Promise<boolean> {
  if (a.length !== b.length) return false;

  const key = await getHmacKey();
  const enc = new TextEncoder();

  const [macA, macB] = await Promise.all([
    crypto.subtle.sign("HMAC", key, enc.encode(a)),
    crypto.subtle.sign("HMAC", key, enc.encode(b)),
  ]);

  const viewA = new Uint8Array(macA);
  const viewB = new Uint8Array(macB);

  let diff = 0;
  for (let i = 0; i < viewA.length; i++) {
    diff |= viewA[i] ^ viewB[i];
  }
  return diff === 0;
}

/**
 * Verify a cron request's CRON_SECRET bearer token.
 * Returns true if valid.
 */
export async function verifyCronSecret(
  authHeader: string | null
): Promise<boolean> {
  const secret = process.env.CRON_SECRET;
  if (!secret || !authHeader) return false;
  return constantTimeEqual(authHeader, `Bearer ${secret}`);
}
