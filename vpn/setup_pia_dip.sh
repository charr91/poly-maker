#!/bin/bash
# Setup script for PIA Dedicated IP with gluetun
# This script generates an OpenVPN config file for your dedicated IP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/pia_dip.ovpn"
CA_FILE="$SCRIPT_DIR/ca.rsa.4096.crt"

# Load environment variables from .env if it exists
if [ -f "$SCRIPT_DIR/../.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
fi

# Check required environment variables
if [ -z "$PIA_USER" ]; then
    echo "Error: PIA_USER not set. Add it to .env or export it."
    echo "This is your PIA username (p-number, e.g., p1234567)"
    exit 1
fi

if [ -z "$PIA_PASS" ]; then
    echo "Error: PIA_PASS not set. Add it to .env or export it."
    exit 1
fi

if [ -z "$PIA_DIP_TOKEN" ]; then
    echo "Error: PIA_DIP_TOKEN not set. Add it to .env or export it."
    echo "This is your dedicated IP token (e.g., DIP2MEj...)"
    exit 1
fi

# Check for jq
if ! command -v jq &>/dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install with: sudo apt install jq"
    exit 1
fi

echo "=== PIA Dedicated IP Setup ==="
echo ""

# Step 1: Get PIA authentication token
echo "Authenticating with PIA..."
TOKEN_RESPONSE=$(curl -s --max-time 15 \
    --location --request POST \
    'https://www.privateinternetaccess.com/api/client/v2/token' \
    --form "username=$PIA_USER" \
    --form "password=$PIA_PASS")

PIA_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.token // empty')

if [ -z "$PIA_TOKEN" ]; then
    echo "Error: Failed to authenticate with PIA"
    echo "Response: $TOKEN_RESPONSE"
    exit 1
fi

echo "Authentication successful!"
echo ""

# Step 2: Get dedicated IP server details
echo "Getting dedicated IP server information..."
DIP_RESPONSE=$(curl -s --max-time 15 \
    --location --request POST \
    'https://www.privateinternetaccess.com/api/client/v2/dedicated_ip' \
    --header 'Content-Type: application/json' \
    --header "Authorization: Token $PIA_TOKEN" \
    --data-raw "{\"tokens\":[\"$PIA_DIP_TOKEN\"]}")

# Check if the DIP token is active
DIP_STATUS=$(echo "$DIP_RESPONSE" | jq -r '.[0].status // empty')
if [ "$DIP_STATUS" != "active" ]; then
    echo "Error: Dedicated IP token is not active"
    echo "Response: $DIP_RESPONSE"
    exit 1
fi

# Parse the response
DIP_IP=$(echo "$DIP_RESPONSE" | jq -r '.[0].ip')
DIP_CN=$(echo "$DIP_RESPONSE" | jq -r '.[0].cn')
DIP_EXPIRE=$(echo "$DIP_RESPONSE" | jq -r '.[0].dip_expire')

if [ -z "$DIP_IP" ] || [ "$DIP_IP" == "null" ]; then
    echo "Could not parse dedicated IP from response"
    echo "Response: $DIP_RESPONSE"
    exit 1
fi

echo "Dedicated IP: $DIP_IP"
echo "Server CN: $DIP_CN"
if [ -n "$DIP_EXPIRE" ] && [ "$DIP_EXPIRE" != "null" ]; then
    echo "Expires: $(date -d @$DIP_EXPIRE 2>/dev/null || echo $DIP_EXPIRE)"
fi

# Generate OpenVPN config with embedded CA certificate
echo ""
echo "Generating OpenVPN configuration..."

# Create auth credentials for dedicated IP
# For dedicated IP, PIA uses: dedicated_ip_$DIP_TOKEN
DIP_CRED="dedicated_ip_$PIA_DIP_TOKEN"
AUTH_FILE="$SCRIPT_DIR/auth.conf"

# Split credential into two lines (PIA format)
echo "${DIP_CRED:0:62}" > "$AUTH_FILE"
echo "${DIP_CRED:62}" >> "$AUTH_FILE"
chmod 600 "$AUTH_FILE"

cat > "$CONFIG_FILE" << 'EOFCONFIG'
client
dev tun
resolv-retry infinite
nobind
persist-key
persist-tun
cipher aes-128-cbc
auth sha1
tls-client
remote-cert-tls server

auth-user-pass /gluetun/auth.conf
compress
verb 1
reneg-sec 0

<ca>
-----BEGIN CERTIFICATE-----
MIIFqzCCBJOgAwIBAgIJAKZ7D5Yv87qDMA0GCSqGSIb3DQEBDQUAMIHoMQswCQYD
VQQGEwJVUzELMAkGA1UECBMCQ0ExEzARBgNVBAcTCkxvc0FuZ2VsZXMxIDAeBgNV
BAoTF1ByaXZhdGUgSW50ZXJuZXQgQWNjZXNzMSAwHgYDVQQLExdQcml2YXRlIElu
dGVybmV0IEFjY2VzczEgMB4GA1UEAxMXUHJpdmF0ZSBJbnRlcm5ldCBBY2Nlc3Mx
IDAeBgNVBCkTF1ByaXZhdGUgSW50ZXJuZXQgQWNjZXNzMS8wLQYJKoZIhvcNAQkB
FiBzZWN1cmVAcHJpdmF0ZWludGVybmV0YWNjZXNzLmNvbTAeFw0xNDA0MTcxNzM1
MThaFw0zNDA0MTIxNzM1MThaMIHoMQswCQYDVQQGEwJVUzELMAkGA1UECBMCQ0Ex
EzARBgNVBAcTCkxvc0FuZ2VsZXMxIDAeBgNVBAoTF1ByaXZhdGUgSW50ZXJuZXQg
QWNjZXNzMSAwHgYDVQQLExdQcml2YXRlIEludGVybmV0IEFjY2VzczEgMB4GA1UE
AxMXUHJpdmF0ZSBJbnRlcm5ldCBBY2Nlc3MxIDAeBgNVBCkTF1ByaXZhdGUgSW50
ZXJuZXQgQWNjZXNzMS8wLQYJKoZIhvcNAQkBFiBzZWN1cmVAcHJpdmF0ZWludGVy
bmV0YWNjZXNzLmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAPXD
L1L9tX6DGf36liA7UBTy5I869z0UVo3lImfOs/GSiFKPtInlesP65577nd7UNzzX
lH/P/CnFPdBWlLp5ze3HRBCc/Avgr5CdMRkEsySL5GHBZsx6w2cayQ2EcRhVTwWp
cdldeNO+pPr9rIgPrtXqT4SWViTQRBeGM8CDxAyTopTsobjSiYZCF9Ta1gunl0G/
8Vfp+SXfYCC+ZzWvP+L1pFhPRqzQQ8k+wMZIovObK1s+nlwPaLyayzw9a8sUnvWB
/5rGPdIYnQWPgoNlLN9HpSmsAcw2z8DXI9pIxbr74cb3/HSfuYGOLkRqrOk6h4RC
OfuWoTrZup1uEOn+fw8CAwEAAaOCAVQwggFQMB0GA1UdDgQWBBQv63nQ/pJAt5tL
y8VJcbHe22ZOsjCCAR8GA1UdIwSCARYwggESgBQv63nQ/pJAt5tLy8VJcbHe22ZO
sqGB7qSB6zCB6DELMAkGA1UEBhMCVVMxCzAJBgNVBAgTAkNBMRMwEQYDVQQHEwpM
b3NBbmdlbGVzMSAwHgYDVQQKExdQcml2YXRlIEludGVybmV0IEFjY2VzczEgMB4G
A1UECxMXUHJpdmF0ZSBJbnRlcm5ldCBBY2Nlc3MxIDAeBgNVBAMTF1ByaXZhdGUg
SW50ZXJuZXQgQWNjZXNzMSAwHgYDVQQpExdQcml2YXRlIEludGVybmV0IEFjY2Vz
czEvMC0GCSqGSIb3DQEJARYgc2VjdXJlQHByaXZhdGVpbnRlcm5ldGFjY2Vzcy5j
b22CCQCmew+WL/O6gzAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEBDQUAA4IBAQAn
a5PgrtxfwTumD4+3/SYvwoD66cB8IcK//h1mCzAduU8KgUXocLx7QgJWo9lnZ8xU
ryXvWab2usg4fqk7FPi00bED4f4qVQFVfGfPZIH9QQ7/48bPM9RyfzImZWUCenK3
7pdw4Bvgoys2rHLHbGen7f28knT2j/cbMxd78tQc20TIObGjo8+ISTRclSTRBtyC
GohseKYpTS9himFERpUgNtefvYHbn70mIOzfOJFTVqfrptf9jXa9N8Mpy3ayfodz
1wiqdteqFXkTYoSDctgKMiZ6GdocK9nMroQipIQtpnwd4yBDWIyC6Bvlkrq5TQUt
YDQ8z9v+DMO6iwyIDRiU
-----END CERTIFICATE-----
</ca>

disable-occ
EOFCONFIG

# Append the remote line with the dedicated IP
echo "remote $DIP_IP 1198 udp" >> "$CONFIG_FILE"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Generated:"
echo "  - $CONFIG_FILE"
echo "  - $AUTH_FILE"
echo ""
echo "Your dedicated IP: $DIP_IP"
echo ""
echo "Next step: docker-compose up -d"
echo ""
