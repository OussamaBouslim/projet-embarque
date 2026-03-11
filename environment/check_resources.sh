#!/bin/bash
# =============================================================
# environment/check_resources.sh
# Vérifie les ressources RAM/CPU des 3 VM Docker
# Usage : bash environment/check_resources.sh
# =============================================================

echo "=================================================="
echo "  VÉRIFICATION DES RESSOURCES — 3 VM Docker"
echo "=================================================="

# Vérifier que Docker est lancé
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker n'est pas lancé !"
    echo "   → Lance Docker Desktop d'abord"
    exit 1
fi

# Vérifier que les VM tournent
VMS=("vm1_capteur_iot" "vm2_gateway_iot" "vm3_edge_server")
for vm in "${VMS[@]}"; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${vm}$"; then
        echo "⚠️  $vm n'est pas lancée → lance : docker-compose up -d"
    fi
done

echo ""
echo "📊 Statistiques en temps réel :"
echo "--------------------------------------------------"
docker stats --no-stream --format \
    "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
    vm1_capteur_iot vm2_gateway_iot vm3_edge_server

echo ""
echo "✅ Vérification terminée"
