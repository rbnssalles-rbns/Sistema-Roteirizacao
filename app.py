#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Nesta seção definimos:
# - Frota de veículos e suas capacidades
# - Catálogo de produtos (SKUs)
# - Clientes fictícios com coordenadas e janelas de recebimento
# - Funções auxiliares para cálculo de distância (Haversine)
# Imports e setup

import math, random, datetime as dt
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List

random.seed(42); np.random.seed(42)

# Datas alvo
DATES = pd.to_datetime(["2025-11-25","2025-11-26","2025-11-27","2025-11-28"])

# Centro de distribuição (CD)
DEPOT = {"id":"CD","lat":-3.7327,"lon":-38.5260}

# Frota
@dataclass
class VehicleType:
    name:str; vol_m3:float; weight_kg:float

FLEET = [
    VehicleType("Pequeno_1",10,2000), VehicleType("Pequeno_2",10,2000),
    VehicleType("Medio_1",18,4000), VehicleType("Medio_2",18,4000),
    VehicleType("Grande_1",24,6000), VehicleType("Grande_2",24,6000)
]

# Catálogo de SKUs
CATALOG = pd.DataFrame([
    {"sku":"PET_2L","unit_volume_m3":0.003,"unit_weight_kg":2.2},
    {"sku":"Caixa_12x1L","unit_volume_m3":0.018,"unit_weight_kg":13.2},
    {"sku":"Bombona_20L","unit_volume_m3":0.035,"unit_weight_kg":21.5}
])

# Parâmetros operacionais
STACKING_EFFICIENCY=0.9; SERVICE_MINUTES=10
URBAN_SPEED_KMH=25; STREET_FACTOR=1.25
WORK_START=dt.time(8,0); WORK_END=dt.time(17,0)
# Aqui criamos:
# - Clientes fictícios (~120) com coordenadas próximas a Fortaleza
# - Dias preferidos de recebimento (Ter–Sex)
# - Janelas de horário de atendimento
# - Pedidos simulados para cada dia, com quantidades aleatórias de SKUs

# Funções auxiliares
def random_point_around_fortaleza(n):
    lat, lon = DEPOT["lat"], DEPOT["lon"]
    return [
        (lat + np.random.uniform(-0.18, 0.18),
         lon + np.random.uniform(-0.18, 0.18))
        for _ in range(n)
    ]

def random_receive_days():
    options = ["Tue", "Wed", "Thu", "Fri"]
    k = np.random.choice([1, 2], p=[0.6, 0.4])
    return sorted(list(np.random.choice(options, k, replace=False)))

def random_time_window():
    start = np.random.choice([8, 9, 10])
    end = np.random.choice([15, 16, 17])
    if end <= start:
        end = start + 6
    return dt.time(start, 0), dt.time(end, 0)

# Clientes
N_CLIENTS = 120
coords = random_point_around_fortaleza(N_CLIENTS)
clients = []
for i, (lat, lon) in enumerate(coords, 1):
    tw_start, tw_end = random_time_window()
    clients.append({
        "client_id": f"C{i:03d}",
        "name": f"Cliente_{i:03d}",
        "lat": lat,
        "lon": lon,
        "preferred_days": random_receive_days(),
        "tw_start": tw_start,
        "tw_end": tw_end
    })
CLIENTS = pd.DataFrame(clients)

# Pedidos
def generate_daily_orders(dates, clients_df, catalog):
    orders = []
    for date in dates:
        dow = date.day_name()[:3]  # Tue, Wed, Thu, Fri
        for _, row in clients_df.iterrows():
            if dow in row["preferred_days"]:
                sku_counts = {
                    "PET_2L": np.random.poisson(30),
                    "Caixa_12x1L": np.random.poisson(6),
                    "Bombona_20L": np.random.poisson(2)
                }
                if sum(sku_counts.values()) == 0:
                    continue
                orders.append({
                    "date": date,
                    "client_id": row["client_id"],
                    "sku_qty": sku_counts
                })
    return pd.DataFrame(orders)

ORDERS = generate_daily_orders(DATES, CLIENTS, CATALOG)
# Calculamos:
# - Volume e peso de cada pedido
# - Validação de viabilidade (se cabe em pelo menos um veículo)

CAT = {r["sku"]: r for _, r in CATALOG.iterrows()}

def order_volume_weight(sku_qty: Dict[str, int]) -> Tuple[float, float]:
    vol = sum(CAT[sku]["unit_volume_m3"] * qty for sku, qty in sku_qty.items()) / STACKING_EFFICIENCY
    wt  = sum(CAT[sku]["unit_weight_kg"] * qty for sku, qty in sku_qty.items())
    return vol, wt

def enrich_orders(orders_df: pd.DataFrame, clients_df: pd.DataFrame) -> pd.DataFrame:
    df = orders_df.copy()
    df[["volume_m3", "weight_kg"]] = df["sku_qty"].apply(lambda q: pd.Series(order_volume_weight(q)))
    return df.merge(
        clients_df[["client_id", "lat", "lon", "tw_start", "tw_end"]],
        on="client_id"
    )

DAILY = enrich_orders(ORDERS, CLIENTS)

max_vol = max(v.vol_m3 for v in FLEET)
max_wt  = max(v.weight_kg for v in FLEET)
DAILY   = DAILY[(DAILY["volume_m3"] <= max_vol) & (DAILY["weight_kg"] <= max_wt)]
# Geramos matriz de distâncias e tempos entre CD e clientes do dia.

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371  # raio da Terra em km
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_costs_for_day(day_df):
    # Lista de nós: CD + clientes
    nodes = [{"id": "CD", "lat": DEPOT["lat"], "lon": DEPOT["lon"]}]
    for _, r in day_df.iterrows():
        nodes.append({"id": r["client_id"], "lat": r["lat"], "lon": r["lon"]})

    N = len(nodes)
    dist = np.zeros((N, N))
    time = np.zeros((N, N))

    # Calcula apenas metade superior da matriz e espelha
    for i in range(N):
        for j in range(i + 1, N):
            d = haversine_km(nodes[i]["lat"], nodes[i]["lon"], nodes[j]["lat"], nodes[j]["lon"]) * STREET_FACTOR
            dist[i, j] = dist[j, i] = d
            time[i, j] = time[j, i] = d / URBAN_SPEED_KMH

    return nodes, dist, time
# Implementamos o algoritmo de Clarke-Wright Savings adaptado para considerar
# capacidade volumétrica e de peso dos veículos. Após gerar as rotas iniciais,
# aplicamos um refinamento simples (2-opt) para reduzir a distância total.

@dataclass
class Demand:
    vol: float
    wt: float
    tw_start: dt.time
    tw_end: dt.time
    service_h: float = SERVICE_MINUTES / 60

def clarke_wright(day_df: pd.DataFrame, dist_km: np.ndarray, time_h: np.ndarray, vehicles: List[VehicleType]):
    N = dist_km.shape[0]
    clients_idx = list(range(1, N))

    # Demandas por cliente
    demands = {
        i: Demand(r["volume_m3"], r["weight_kg"], r["tw_start"], r["tw_end"])
        for i, (_, r) in enumerate(day_df.iterrows(), start=1)
    }

    # Inicial: uma rota CD-i-CD para cada cliente
    routes = [[0, i, 0] for i in clients_idx]
    route_loads = {i: (demands[i].vol, demands[i].wt, demands[i].service_h) for i in clients_idx}

    # Savings
    savings = [
        (dist_km[0, i] + dist_km[0, j] - dist_km[i, j], i, j)
        for i in clients_idx for j in clients_idx if i < j
    ]
    savings.sort(reverse=True, key=lambda x: x[0])

    max_vol = max(v.vol_m3 for v in vehicles)
    max_wt = max(v.weight_kg for v in vehicles)

    node2route = {i: k for k, i in enumerate(clients_idx)}
    route_cargo = {k: route_loads[route[1]] for k, route in enumerate(routes)}

    def route_ends(route): return route[1], route[-2]

    # Merge rotas
    for _, i, j in savings:
        ri, rj = node2route[i], node2route[j]
        if ri == rj: continue
        route_i, route_j = routes[ri], routes[rj]
        i_start, i_end = route_ends(route_i)
        j_start, j_end = route_ends(route_j)
        new_route = None

        if i_start == i and j_start == j:
            new_route = route_j[::-1][:-1] + route_i[1:]
        elif i_start == i and j_end == j:
            new_route = route_j[:-1] + route_i[1:]
        elif i_end == i and j_start == j:
            new_route = route_i[:-1] + route_j[1:]
        elif i_end == i and j_end == j:
            new_route = route_i[:-1] + route_j[::-1][1:]

        if new_route:
            vol_i, wt_i, svc_i = route_cargo[ri]
            vol_j, wt_j, svc_j = route_cargo[rj]
            vol_new, wt_new, svc_new = vol_i + vol_j, wt_i + wt_j, svc_i + svc_j
            if vol_new <= max_vol and wt_new <= max_wt:
                routes[ri] = new_route
                route_cargo[ri] = (vol_new, wt_new, svc_new)
                for node in route_j[1:-1]:
                    node2route[node] = ri
                routes[rj] = []
                route_cargo[rj] = (0, 0, 0)

    routes = [r for r in routes if r]

    # 2-opt
    def route_distance(route): return sum(dist_km[route[k], route[k+1]] for k in range(len(route)-1))
    def two_opt(route):
        best = route[:]
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_distance(new_route) < route_distance(best):
                    best = new_route
        return best

    routes_opt = [two_opt(r) for r in routes]

    # Cargas finais
    route_cargo_final = [
        (sum(demands[n].vol for n in r[1:-1]),
         sum(demands[n].wt for n in r[1:-1]),
         sum(demands[n].service_h for n in r[1:-1]))
        for r in routes_opt
    ]

    return routes_opt, route_cargo_final

# Atribuição de veículos
def assign_vehicles_to_routes(routes, cargo, vehicles, dist_km):
    order = sorted(range(len(routes)), key=lambda k: cargo[k][0], reverse=True)
    available = vehicles[:]
    assignments = []
    for k in order:
        vol, wt, _ = cargo[k]
        feasible = [v for v in available if v.vol_m3 >= vol and v.weight_kg >= wt]
        if feasible:
            chosen = sorted(feasible, key=lambda v: (v.vol_m3, v.weight_kg))[0]
            assignments.append((k, chosen))
            available.remove(chosen)
        else:
            assignments.append((k, None))
    dists = [sum(dist_km[r[i], r[i+1]] for i in range(len(r)-1)) for r in routes]
    return assignments, dists
# Nesta seção simulamos cada dia (25 a 28/11/2025), geramos as rotas com a heurística,
# atribuimos veículos e mostramos um resumo com distância total, número de rotas e
# eventuais violações de janelas de atendimento.
#Bloco 6
# =========================
# Parte 1 — Funções de negócio
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from io import BytesIO

def simulate_day(date: pd.Timestamp, daily_df: pd.DataFrame):
    day_orders = daily_df[daily_df["date"] == date].reset_index(drop=True)
    if day_orders.empty:
        return {
            "date": date,
            "routes": [],
            "cargo": [],
            "assign": [],
            "dists": [],
            "nodes": [],
            "dist_km": None,
            "violations": [],
            "orders": day_orders,
            "anomalies": [],
            "unrouted_clients": [],
        }

    nodes, dist_km, time_h = build_costs_for_day(day_orders)
    routes, cargo = clarke_wright(day_orders, dist_km, time_h, FLEET)
    assignments, dists = assign_vehicles_to_routes(routes, cargo, FLEET, dist_km)

    start_dt = dt.datetime.combine(date.date(), WORK_START)
    violations = []
    for idx_r, r in enumerate(routes):
        t = start_dt
        for node in r[1:-1]:
            prev = r[r.index(node) - 1]
            travel_h = time_h[prev, node]
            t += dt.timedelta(hours=travel_h)
            ord_row = day_orders.iloc[node - 1]
            if not (ord_row["tw_start"] <= t.time() <= ord_row["tw_end"]):
                violations.append({
                    "route": idx_r,
                    "client_id": ord_row["client_id"],
                    "arrival": t.time(),
                    "tw": (ord_row["tw_start"], ord_row["tw_end"]),
                })
            t += dt.timedelta(hours=SERVICE_MINUTES / 60)

    return {
        "date": date,
        "routes": routes,
        "cargo": cargo,
        "assign": assignments,
        "dists": dists,
        "nodes": nodes,
        "dist_km": dist_km,
        "violations": violations,
        "orders": day_orders,
        "anomalies": [],
        "unrouted_clients": [],
    }

def daily_summary(result):
    date = result["date"]
    total_dist = sum(result["dists"]) if result["dists"] else 0
    used = sum(1 for _, v in result["assign"] if v is not None)
    st.subheader(f"📅 {date.date()}")
    st.write(
        f"Rotas: {len(result['routes'])} | Veículos usados: {used} | "
        f"Distância total: {total_dist:.1f} km"
    )
    for i, r in enumerate(result["routes"]):
        vol, wt, svc = result["cargo"][i]
        matches = [v for k, v in result["assign"] if k == i]
        assign = matches[0] if matches else None
        vname = assign.name if assign else "SEM_VEICULO"
        dist = result["dists"][i] if i < len(result["dists"]) else 0.0
        stops = len(r) - 2

        if assign and assign.vol_m3 and assign.weight_kg:
            eficiencia_vol = vol / assign.vol_m3 * 100
            eficiencia_wt = wt / assign.weight_kg * 100
            eficiencia_txt = f" | Ocupação: {eficiencia_vol:.0f}% vol / {eficiencia_wt:.0f}% peso"
        else:
            eficiencia_txt = ""

        marcador = " 🔸 ANÔMALA" if stops < 10 else ""
        st.write(
            f"- Rota {i}: veículo={vname}, paradas={stops}, dist={dist:.1f} km, "
            f"carga={vol:.2f} m³ / {wt:.0f} kg{eficiencia_txt}{marcador}"
        )

    if result.get("violations"):
        st.warning(f"⚠️ Janelas violadas: {len(result['violations'])}")
    else:
        st.success("✅ Sem violações de janela (simulação simples).")

def plot_routes_by_vehicle(result, route_index, figsize=(7, 7)):
    if not result["routes"]:
        st.write("Sem rotas para o dia.")
        return
    if route_index < 0 or route_index >= len(result["routes"]):
        st.warning(f"Índice de rota inválido: {route_index}")
        return

    r = result["routes"][route_index]
    day_orders = result["orders"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(DEPOT["lon"], DEPOT["lat"], c="red", s=80, label="CD")

    xs, ys = [], []
    for node in r:
        if node == 0:
            xs.append(DEPOT["lon"]); ys.append(DEPOT["lat"])
        else:
            row = day_orders.iloc[node - 1]
            xs.append(row["lon"]); ys.append(row["lat"])
            ax.scatter(row["lon"], row["lat"], c="blue", s=30, alpha=0.7)

    ax.plot(xs, ys, color="gray", linewidth=2)
    ax.set_title(f"Rota {route_index} – {result['date'].date()}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="best"); ax.grid(True)
    st.pyplot(fig)

def rebalance_routes(res, min_clients=10, max_clients=25):
    rotas_final = []
    cargo_final = []
    dispersos_rows = []
    anomalies_labels = []
    unrouted_ids = []

    for i, r in enumerate(res["routes"]):
        stops = len(r) - 2
        if stops > max_clients:
            blocos = [r[1:-1][j:j+max_clients] for j in range(0, stops, max_clients)]
            for bloco in blocos:
                nova_rota = [0] + bloco + [0]
                rotas_final.append(nova_rota)
                vol = sum(res["orders"].iloc[node-1]["volume_m3"] for node in bloco)
                wt  = sum(res["orders"].iloc[node-1]["weight_kg"] for node in bloco)
                svc = sum(SERVICE_MINUTES/60 for _ in bloco)
                cargo_final.append((vol, wt, svc))
        elif stops < min_clients:
            for node in r[1:-1]:
                dispersos_rows.append(res["orders"].iloc[node-1])
        else:
            rotas_final.append(r)
            cargo_final.append(res["cargo"][i])

    if dispersos_rows:
        pedidos_rebalancear = pd.DataFrame(dispersos_rows)
        nodes_rb, dist_km_rb, time_h_rb = build_costs_for_day(pedidos_rebalancear)
        novas_rotas, novas_cargas = clarke_wright(pedidos_rebalancear, dist_km_rb, time_h_rb, FLEET)
        for nr, nc in zip(novas_rotas, novas_cargas):
            stops_nr = len(nr) - 2
            if stops_nr >= min_clients:
                rotas_final.append(nr)
                cargo_final.append(nc)
            else:
                ids = [pedidos_rebalancear.iloc[node-1]["client_id"] for node in nr[1:-1]]
                anomalies_labels.append(f"Rota anômala com {stops_nr} clientes: IDs {', '.join(ids)}")
                unrouted_ids.extend(ids)
                rotas_final.append(nr)
                cargo_final.append(nc)

    assignments_final, dists_final = assign_vehicles_to_routes(rotas_final, cargo_final, FLEET, res["dist_km"])

    violations = []
    start_dt = dt.datetime.combine(res["date"].date(), WORK_START)
    for idx_r, r in enumerate(rotas_final):
        t = start_dt
        for node in r[1:-1]:
            prev = r[r.index(node) - 1]
            try:
                travel_h = res["dist_km"][prev, node]
            except Exception:
                travel_h = 0.0
            t += dt.timedelta(hours=travel_h)
            ord_row = res["orders"].iloc[node - 1]
            if not (ord_row["tw_start"] <= t.time() <= ord_row["tw_end"]):
                violations.append({
                    "route": idx_r,
                    "client_id": ord_row["client_id"],
                    "arrival": t.time(),
                    "tw": (ord_row["tw_start"], ord_row["tw_end"]),
                })
            t += dt.timedelta(hours=SERVICE_MINUTES / 60)

    return {
        "date": res["date"],
        "routes": rotas_final,
        "cargo": cargo_final,
        "assign": assignments_final,
        "dists": dists_final,
        "nodes": res["nodes"],
        "dist_km": res["dist_km"],
        "violations": violations,
        "orders": res["orders"],
        "anomalies": anomalies_labels,
        "unrouted_clients": unrouted_ids,
    }
# Parte 2 — Interface principal
st.title("🚚 Sistema de Roteirização de Entregas")

# Estado inicial
if "res" not in st.session_state:
    st.session_state.res = simulate_day(DATES[0], DAILY)
if "rotas_selecionadas" not in st.session_state:
    st.session_state.rotas_selecionadas = []
if "res_base" not in st.session_state:
    st.session_state.res_base = st.session_state.res  # última versão oficial
if "pedidos_extras" not in st.session_state:
    st.session_state.pedidos_extras = []
if "before_res" not in st.session_state:
    st.session_state.before_res = None
if "after_res" not in st.session_state:
    st.session_state.after_res = None
if "toast_success" not in st.session_state:
    st.session_state.toast_success = None

# Seleção de dia
dia_selecionado = st.selectbox("Selecione o dia", DATES, key="sel_dia_principal")

# Simulação para o dia selecionado
st.session_state.res = simulate_day(dia_selecionado, DAILY)
res = st.session_state.res

# Resumo
daily_summary(res)

# Filtro de rotas
rotas_disponiveis = [f"Rota {i}" for i in range(len(res["routes"]))]
if not st.session_state.rotas_selecionadas:
    st.session_state.rotas_selecionadas = rotas_disponiveis[:]

rotas_selecionadas = st.multiselect(
    "Escolha as rotas para visualizar",
    rotas_disponiveis,
    default=st.session_state.rotas_selecionadas,
    key="ms_rotas_principal",
)
st.session_state.rotas_selecionadas = rotas_selecionadas

# Mapas
if st.checkbox("Mostrar mapas das rotas", key="cb_mapas_principal"):
    for i in range(len(res["routes"])):
        if f"Rota {i}" in st.session_state.rotas_selecionadas:
            plot_routes_by_vehicle(res, i, figsize=(7, 7))

# Botão de reprocessar
if st.button("🔄 Reprocessar rotas", key="btn_reprocessar_principal"):
    res_raw = simulate_day(dia_selecionado, DAILY)
    res_bal = rebalance_routes(res_raw, min_clients=10, max_clients=25)

    st.session_state.res = res_bal
    st.session_state.res_base = res_bal
    res = st.session_state.res

    st.success("Rotas reprocessadas e rebalanceadas com sucesso!")
    daily_summary(res)

    rotas_disponiveis = [f"Rota {i}" for i in range(len(res["routes"]))]
    st.session_state.rotas_selecionadas = [
        r for r in st.session_state.rotas_selecionadas if r in rotas_disponiveis
    ] or rotas_disponiveis[:]

    if st.checkbox("Mostrar mapas das rotas (após reprocessar)", key="cb_mapas_pos_reproc"):
        for i in range(len(res["routes"])): 
            if f"Rota {i}" in st.session_state.rotas_selecionadas:
                plot_routes_by_vehicle(res, i, figsize=(7, 7))
# Parte 3 — Planejamento D+1 e pedidos extras
st.header("📆 Planejamento Rolling (D+1) + Pedidos Extras")

def estimate_volume_weight(sku_qty: dict):
    vol, wt = 0.0, 0.0
    for sku, qty in sku_qty.items():
        row = CATALOG[CATALOG["sku"] == sku]
        if not row.empty:
            vol += row["unit_volume_m3"].values[0] * qty
            wt += row["unit_weight_kg"].values[0] * qty
    vol = vol / STACKING_EFFICIENCY
    return vol, wt

def add_last_minute_order(date: pd.Timestamp, client_id: str, sku_qty: dict, daily_df: pd.DataFrame):
    new = {"date": date, "client_id": client_id, "sku_qty": sku_qty}
    df = pd.concat([daily_df[["date","client_id","sku_qty"]], pd.DataFrame([new])], ignore_index=True)
    df = enrich_orders(df, CLIENTS)
    return df

def adicionar_pedido_extra_e_replanejar(date_sel: pd.Timestamp, client_id: str, sku_qty: dict):
    res_base = st.session_state.res_base if "res_base" in st.session_state else simulate_day(date_sel, DAILY)
    res_original = simulate_day(date_sel, DAILY)

    vol_extra, wt_extra = estimate_volume_weight(sku_qty)
    svc_extra = SERVICE_MINUTES / 60

    # Localiza rota do cliente na base
    rota_idx = None
    for i, r in enumerate(res_base["routes"]):
        for node in r[1:-1]:
            if res_base["orders"].iloc[node - 1]["client_id"] == client_id:
                rota_idx = i
                break
        if rota_idx is not None:
            break

    # Atualiza DF com pedido extra
    df_mod = add_last_minute_order(date_sel, client_id, sku_qty, DAILY)
    res_mod_matriz = simulate_day(date_sel, df_mod)

    if rota_idx is None:
        # Cliente não estava em nenhuma rota → rebalanceia
        res_mod = rebalance_routes(res_mod_matriz, min_clients=10, max_clients=25)
        st.session_state.res = res_mod
        st.session_state.res_base = res_mod
        return res_original, res_mod

    # Verifica capacidade do veículo atual
    vol_atual, wt_atual, svc_atual = res_base["cargo"][rota_idx]
    matches = [v for k, v in res_base["assign"] if k == rota_idx]
    assign_v = matches[0] if matches else None

    if assign_v and (vol_atual + vol_extra <= assign_v.vol_m3) and (wt_atual + wt_extra <= assign_v.weight_kg):
        # Atualiza carga mantendo a rota
        res_base["cargo"][rota_idx] = (vol_atual + vol_extra, wt_atual + wt_extra, svc_atual + svc_extra)

        # Recalcula violações para a rota afetada usando matriz recalculada
        nodes_new, dist_km_new, time_h_new = build_costs_for_day(res_mod_matriz["orders"])
        start_dt = dt.datetime.combine(date_sel.date(), WORK_START)
        t = start_dt
        violacoes = []
        rota = res_base["routes"][rota_idx]
        for node in rota[1:-1]:
            prev = rota[rota.index(node) - 1]
            try:
                travel_h = time_h_new[prev, node]
            except Exception:
                travel_h = 0.0
            t += dt.timedelta(hours=travel_h)
            ord_row = res_mod_matriz["orders"].iloc[node - 1]
            if not (ord_row["tw_start"] <= t.time() <= ord_row["tw_end"]):
                violacoes.append({
                    "route": rota_idx,
                    "client_id": ord_row["client_id"],
                    "arrival": t.time(),
                    "tw": (ord_row["tw_start"], ord_row["tw_end"]),
                })
            t += dt.timedelta(hours=SERVICE_MINUTES / 60)

        res_base["violations"] = [v for v in res_base["violations"] if v["route"] != rota_idx] + violacoes
        st.session_state.res = res_base
        st.session_state.res_base = res_base
        return res_original, res_base

    # Caso não caiba → rebalanceia e reatribui
    res_mod = rebalance_routes(res_mod_matriz, min_clients=10, max_clients=25)
    st.session_state.res = res_mod
    st.session_state.res_base = res_mod
    return res_original, res_mod

# D+1 considerando extras acumulados
data_corrente_rolling = st.selectbox("Selecione a data corrente", DATES, key="sel_dia_rolling")
if st.button("Planejar dia seguinte (D+1)", key="btn_plan_next_rolling"):
    idx = list(DATES).index(data_corrente_rolling)
    if idx >= len(DATES) - 1:
        st.warning("Não há dia seguinte dentro do período.")
    else:
        next_date = DATES[idx + 1]
        df_base = DAILY.copy()
        for pedido in st.session_state.pedidos_extras:
            if pd.to_datetime(pedido["date"]) == next_date:
                df_base = add_last_minute_order(pedido["date"], pedido["client_id"], pedido["sku_qty"], df_base)

        res_next = simulate_day(next_date, df_base)
        daily_summary(res_next)
        for i in range(len(res_next["routes"])):
            plot_routes_by_vehicle(res_next, i, figsize=(7,7))

# UI de pedidos extras
st.subheader("➕ Adicionar pedido extra e replanejar")

data_extra = st.date_input("Data do pedido extra", min_value=min(DATES), max_value=max(DATES), key="inp_data_extra")
cliente_extra = st.selectbox("Cliente", CLIENTS["client_id"].unique(), key="sel_cliente_extra")

sku_inputs = {}
for sku in CATALOG["sku"].unique():
    qtd = st.number_input(f"{sku}", min_value=0, step=1, key=f"inp_qtd_{sku}_extra")
    if qtd > 0:
        sku_inputs[sku] = qtd

# Botão para adicionar pedido extra e replanejar
if st.button("Adicionar pedido extra e replanejar", key="btn_pedido_extra_exec"):
    if not sku_inputs:
        st.error("Informe ao menos 1 SKU para o pedido extra.")
    else:
        before_res, after_res = adicionar_pedido_extra_e_replanejar(
            pd.to_datetime(data_extra), cliente_extra, sku_inputs
        )

        # Salva no histórico
        st.session_state.pedidos_extras.append({
            "date": pd.to_datetime(data_extra),
            "client_id": cliente_extra,
            "sku_qty": sku_inputs,
        })

        # Guarda comparação e feedback para próxima renderização
        st.session_state.before_res = before_res
        st.session_state.after_res = after_res
        st.session_state.toast_success = f"✅ Pedido extra do cliente {cliente_extra} registrado e replanejado com sucesso!"

        # Reinicia interface para limpar todos os campos
        st.rerun()

# Exibe feedback pós-rerun (se disponível)
if "toast_success" in st.session_state and st.session_state.toast_success:
    st.success(st.session_state.toast_success)
    st.session_state.toast_success = None

# Exibe comparação antes/depois se disponível
if st.session_state.before_res is not None and st.session_state.after_res is not None:
    st.subheader("📊 Comparação antes/depois")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Antes do pedido extra")
        daily_summary(st.session_state.before_res)
    with col2:
        st.caption("Depois do pedido extra")
        daily_summary(st.session_state.after_res)

    for i in range(len(st.session_state.after_res["routes"])):
        plot_routes_by_vehicle(st.session_state.after_res, i, figsize=(7,7))
# Parte 4 — Histórico de pedidos extras, download e semana completa

    # Histórico de pedidos extras + Download Excel
if st.session_state.pedidos_extras:
    st.subheader("📋 Histórico de pedidos extras")
    df_extras = pd.DataFrame(st.session_state.pedidos_extras)
    st.dataframe(df_extras)

    if not df_extras.empty:
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_extras.to_excel(writer, index=False, sheet_name="PedidosExtras")
            output.seek(0)

            st.download_button(
                label="⬇️ Baixar pedidos extras em Excel",
                data=output,
                file_name="pedidos_extras.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_download_extras",
            )
        except ModuleNotFoundError:
            st.error("⚠️ Não foi possível gerar o Excel: instale 'openpyxl' ou declare no requirements.txt.")
    else:
        st.info("Nenhum pedido extra para exportar.")

    if st.button("🗑️ Limpar histórico de pedidos extras", key="btn_limpar_extras"):
        st.session_state.pedidos_extras = []
        st.success("Histórico de pedidos extras limpo.")


# Simular semana completa com extras
st.subheader("📈 Simular toda a semana com pedidos extras")
if st.button("Simular semana completa com extras", key="btn_semana_completa_extras"):
    for d in DATES:
        df_mod = DAILY.copy()
        for pedido in st.session_state.pedidos_extras:
            if pd.to_datetime(pedido["date"]) == d:
                df_mod = add_last_minute_order(pedido["date"], pedido["client_id"], pedido["sku_qty"], df_mod)
        res_dia = simulate_day(d, df_mod)
        st.markdown(f"### 📅 {d.date()}")
        daily_summary(res_dia)
        for i in range(len(res_dia["routes"])):
            plot_routes_by_vehicle(res_dia, i, figsize=(6,6))


# Nesta seção planejamos sempre o dia seguinte a partir de uma data corrente.
# Também simulamos a inclusão de pedidos de última hora e reotimizamos as rotas.
# (Bloco integrado à Parte B do Bloco 6 para evitar duplicidade de interface)

# Se desejar isolar visualmente este bloco em outro arquivo, mantenha as chaves exclusivas:
# - sel_dia_rolling, btn_plan_next_rolling
# - inp_data_extra, sel_cliente_extra, inp_qtd_{sku}_extra, btn_pedido_extra_exec
# - btn_semana_completa_extras
#
# Observação: As funções auxiliares (add_last_minute_order, estimate_volume_weight,
# adicionar_pedido_extra_e_replanejar) já estão definidas na Parte B do Bloco 6.
# Reaproveite-as para evitar duplicidade de código.


# In[ ]:




