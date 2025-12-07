#!/bin/bash
# Poly-Maker Bot Helper Script
# Usage: ./bot.sh [command] [options]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect docker compose command (v2 plugin vs v1 standalone)
if docker compose version &>/dev/null; then
    DC="docker compose"
elif command -v docker-compose &>/dev/null; then
    DC="docker-compose"
else
    echo -e "${RED}Error: Neither 'docker compose' nor 'docker-compose' found${NC}"
    exit 1
fi

# Track which mode is running
get_running_mode() {
    if $DC ps --quiet polybot 2>/dev/null | grep -q .; then
        echo "vpn"
    elif $DC -f docker-compose.novpn.yml ps --quiet polybot 2>/dev/null | grep -q .; then
        echo "novpn"
    elif $DC -f docker-compose.pia.yml ps --quiet polybot 2>/dev/null | grep -q .; then
        echo "pia"
    else
        echo "none"
    fi
}

# Stop all modes
stop_all() {
    $DC down 2>/dev/null || true
    $DC -f docker-compose.novpn.yml down 2>/dev/null || true
    $DC -f docker-compose.pia.yml down 2>/dev/null || true
}

# Get compose file for mode
get_compose_file() {
    case "$1" in
        novpn) echo "docker-compose.novpn.yml" ;;
        pia)   echo "docker-compose.pia.yml" ;;
        *)     echo "docker-compose.yml" ;;
    esac
}

# Show usage
usage() {
    echo "Poly-Maker Bot Helper"
    echo ""
    echo "Usage: ./bot.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start [mode]     Start bot (modes: vpn*, novpn, pia)"
    echo "  stop             Stop bot"
    echo "  restart          Restart bot in current mode"
    echo "  status           Show status and public IP"
    echo "  logs [service]   Follow logs (services: vpn, polybot)"
    echo "  ip               Show current public IP"
    echo "  vpn setup        Regenerate dedicated IP VPN config"
    echo "  vpn region NAME  Set PIA region for 'pia' mode"
    echo "  build            Rebuild bot image"
    echo ""
    echo "Modes:"
    echo "  vpn    - Dedicated IP VPN (default)"
    echo "  novpn  - No VPN, direct connection"
    echo "  pia    - Regular PIA region (set with 'vpn region')"
    echo ""
    echo "Examples:"
    echo "  ./bot.sh start           # Start with dedicated IP VPN"
    echo "  ./bot.sh start novpn     # Start without VPN"
    echo "  ./bot.sh start pia       # Start with regular PIA region"
    echo "  ./bot.sh vpn region 'US Texas'"
    echo "  ./bot.sh logs polybot"
}

case "$1" in
    start)
        MODE="${2:-vpn}"
        COMPOSE_FILE=$(get_compose_file "$MODE")

        if [ ! -f "$COMPOSE_FILE" ]; then
            echo -e "${RED}Error: $COMPOSE_FILE not found${NC}"
            exit 1
        fi

        # Check if dedicated IP mode needs setup
        if [ "$MODE" = "vpn" ] && [ ! -f "vpn/pia_dip.ovpn" ]; then
            echo -e "${YELLOW}VPN config not found. Running setup...${NC}"
            ./vpn/setup_pia_dip.sh
        fi

        echo -e "${GREEN}Starting bot in $MODE mode...${NC}"
        stop_all
        $DC -f "$COMPOSE_FILE" up -d

        echo ""
        echo -e "${GREEN}Bot started!${NC}"
        sleep 2

        # Show IP for VPN modes
        if [ "$MODE" != "novpn" ]; then
            echo -n "Public IP: "
            $DC -f "$COMPOSE_FILE" exec -T polybot curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "(waiting for VPN...)"
        fi
        ;;

    stop)
        echo "Stopping bot..."
        stop_all
        echo -e "${GREEN}Bot stopped.${NC}"
        ;;

    restart)
        MODE=$(get_running_mode)
        if [ "$MODE" = "none" ]; then
            echo -e "${YELLOW}Bot is not running. Use './bot.sh start' to start.${NC}"
            exit 1
        fi
        COMPOSE_FILE=$(get_compose_file "$MODE")
        echo "Restarting bot in $MODE mode..."
        $DC -f "$COMPOSE_FILE" restart
        echo -e "${GREEN}Bot restarted.${NC}"
        ;;

    status)
        MODE=$(get_running_mode)
        echo "Mode: $MODE"
        echo ""

        if [ "$MODE" = "none" ]; then
            echo "Bot is not running."
        else
            COMPOSE_FILE=$(get_compose_file "$MODE")
            $DC -f "$COMPOSE_FILE" ps
            echo ""
            echo -n "Public IP: "
            $DC -f "$COMPOSE_FILE" exec -T polybot curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unavailable"
        fi
        ;;

    logs)
        MODE=$(get_running_mode)
        if [ "$MODE" = "none" ]; then
            echo "Bot is not running."
            exit 1
        fi
        COMPOSE_FILE=$(get_compose_file "$MODE")
        SERVICE="${2:-}"
        $DC -f "$COMPOSE_FILE" logs -f $SERVICE
        ;;

    ip)
        MODE=$(get_running_mode)
        if [ "$MODE" = "none" ]; then
            echo "Bot is not running."
            exit 1
        fi
        COMPOSE_FILE=$(get_compose_file "$MODE")
        $DC -f "$COMPOSE_FILE" exec -T polybot curl -s https://api.ipify.org
        echo ""
        ;;

    vpn)
        case "$2" in
            setup)
                echo "Regenerating dedicated IP VPN config..."
                ./vpn/setup_pia_dip.sh

                MODE=$(get_running_mode)
                if [ "$MODE" = "vpn" ]; then
                    echo ""
                    echo "Restarting VPN with new config..."
                    $DC restart vpn
                fi
                ;;
            region)
                if [ -z "$3" ]; then
                    echo "Usage: ./bot.sh vpn region <REGION>"
                    echo ""
                    echo "Examples:"
                    echo "  ./bot.sh vpn region 'US Texas'"
                    echo "  ./bot.sh vpn region 'CA Vancouver'"
                    echo "  ./bot.sh vpn region 'UK London'"
                    echo "  ./bot.sh vpn region 'DE Berlin'"
                    exit 1
                fi

                # Update PIA_REGION in .env
                REGION="$3"
                if grep -q "^PIA_REGION=" .env 2>/dev/null; then
                    sed -i "s/^PIA_REGION=.*/PIA_REGION=$REGION/" .env
                else
                    echo "PIA_REGION=$REGION" >> .env
                fi
                echo -e "${GREEN}PIA region set to: $REGION${NC}"
                echo "Use './bot.sh start pia' to start with this region."
                ;;
            *)
                echo "VPN commands:"
                echo "  ./bot.sh vpn setup           - Regenerate dedicated IP config"
                echo "  ./bot.sh vpn region <NAME>   - Set PIA region for 'pia' mode"
                ;;
        esac
        ;;

    build)
        echo "Rebuilding bot image..."
        $DC build --no-cache
        echo -e "${GREEN}Build complete.${NC}"
        ;;

    *)
        usage
        ;;
esac
