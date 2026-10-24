#!/usr/bin/env bash
# Local CA + server cert so phones and iPads get a secure context
# (getUserMedia needs one).
#
# Two certs, not one self-signed: Chrome applies a warned-through exception to
# the websocket too, but Safari on iOS/iPadOS never does for wss:// (WebKit bug
# 41419) -- the page loads and the live panel never connects. The iPad has to
# genuinely trust the issuer, and then iOS also enforces Apple's server-cert
# rules: <=825 days, EKU serverAuth, names in SAN. The CA is what gets
# installed on the iPad and is made once; the server cert is reissued here
# whenever the LAN IP moves, without touching the iPad.
#
# Portable: Linux/WSL2 (OpenSSL) and macOS (LibreSSL in /usr/bin/openssl, bash
# 3.2, BSD userland). Extensions go through small config files instead of
# `-addext` / `-ext`, which LibreSSL lacks or only has in newer releases.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl nggak ketemu. Install dulu (macOS: brew install openssl)." >&2
    exit 1
fi

mkdir -p certs
cd certs
umask 077

# Scratch space for the config files; removed on every exit path.
WORK=$(mktemp -d "${TMPDIR:-/tmp}/supermeet-cert.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# Run an openssl command quietly, but show its own error output if it fails
# (the old 2>/dev/null hid the reason, e.g. an unsupported flag).
ossl() {
    if ! openssl "$@" 2>"$WORK/err"; then
        cat "$WORK/err" >&2
        echo "openssl gagal: openssl $1 ..." >&2
        echo "Di macOS coba: brew install openssl, lalu jalankan lagi dengan" >&2
        echo "  PATH=\"\$(brew --prefix openssl)/bin:\$PATH\" ./make-cert.sh" >&2
        return 1
    fi
}

# True if $1 has something other than whitespace in it.
nonblank() { [ -n "$(printf '%s' "$1" | tr -d '[:space:]')" ]; }

# Every IPv4 address this machine answers on, one per line. Tries, in order:
#   - `hostname -I`            Linux / WSL (GNU hostname)
#   - `ipconfig getifaddr enN` macOS (prints the address of one interface)
#   - `ifconfig` parsing        macOS fallback, other BSDs
# Loopback and link-local (169.254.x.x) are dropped.
lan_ips() {
    local out="" i ip
    out=$(hostname -I 2>/dev/null | tr ' ' '\n' || true)
    if ! nonblank "$out" && command -v ipconfig >/dev/null 2>&1; then
        for i in 0 1 2 3 4 5 6 7 8 9; do
            ip=$(ipconfig getifaddr "en$i" 2>/dev/null || true)
            if [ -n "$ip" ]; then out="$out
$ip"; fi
        done
    fi
    if ! nonblank "$out" && command -v ifconfig >/dev/null 2>&1; then
        out=$(ifconfig 2>/dev/null | awk '/^[[:space:]]+inet /{print $2}' |
              sed 's/^addr://' || true)
    fi
    printf '%s\n' "$out" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' |
        grep -vE '^(127\.|169\.254\.)' | awk '!seen[$0]++' || true
}

if [ ! -f ca.pem ] || [ ! -f ca-key.pem ]; then
    cat > "$WORK/ca.cnf" <<'EOF'
[req]
distinguished_name = dn
prompt = no
x509_extensions = v3_ca
[dn]
CN = supermeet local CA
[v3_ca]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
EOF
    ossl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$WORK/ca.cnf" \
        -keyout ca-key.pem -out ca.pem -subj "/CN=supermeet local CA"
    rm -f cert.pem key.pem   # not issued by this CA, whatever it was
    echo "local CA generated: certs/ca.pem"
fi

# IPv4 only. In WSL mirrored mode this lists every Windows adapter, and we
# cannot know which one the iPad reaches us on, so cover them all.
LANS=$(lan_ips)

current() {
    [ -f cert.pem ] && [ -f key.pem ] || return 1
    openssl verify -CAfile ca.pem cert.pem >/dev/null 2>&1 || return 1
    openssl x509 -in cert.pem -noout -checkend $((30 * 86400)) >/dev/null || return 1
    # `-ext subjectAltName` is OpenSSL only; the -text dump works in LibreSSL too.
    local san ip
    san=$(openssl x509 -in cert.pem -noout -text 2>/dev/null |
          grep -A1 'Subject Alternative Name' || true)
    for ip in $LANS; do
        grep -qE "IP Address:${ip//./\\.}(,|[[:space:]]*$)" <<<"$san" || return 1
    done
}
current && exit 0

SAN="DNS:localhost,IP:127.0.0.1"
for ip in $LANS; do SAN="$SAN,IP:$ip"; done

cat > "$WORK/req.cnf" <<'EOF'
[req]
distinguished_name = dn
prompt = no
[dn]
CN = supermeet
EOF
printf '%s\n' \
    "[server_ext]" \
    "subjectAltName=$SAN" \
    "extendedKeyUsage=serverAuth" \
    "keyUsage=critical,digitalSignature,keyEncipherment" \
    "basicConstraints=critical,CA:FALSE" > "$WORK/server.ext"

ossl req -new -newkey rsa:2048 -nodes -config "$WORK/req.cnf" \
    -keyout key.pem -out "$WORK/server.csr" -subj "/CN=supermeet"
ossl x509 -req -in "$WORK/server.csr" -CA ca.pem -CAkey ca-key.pem \
    -set_serial "0x$(openssl rand -hex 16)" -days 820 -out cert.pem \
    -extfile "$WORK/server.ext" -extensions server_ext
echo "server cert issued for localhost $(echo $LANS)"
