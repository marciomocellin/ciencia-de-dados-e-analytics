"""
Módulo para análise de redes de relacionamentos entre sócios

Autor: MVP - Pós-Graduação em Ciência de Dados e Analytics - PUC-Rio
Data: Janeiro/2026
"""

import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import community
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def build_bipartite_graph(df_socios, sample_size=None):
    """
    Constrói grafo bipartido de sócios e empresas
    
    Args:
        df_socios (pd.DataFrame): DataFrame de sócios
        sample_size (int): Tamanho da amostra (None para todos)
        
    Returns:
        nx.Graph: Grafo bipartido
    """
    # Limpar dados
    df = df_socios[['cnpj_basico', 'nome_socio']].copy()
    df = df.dropna(subset=['nome_socio'])
    df = df.drop_duplicates()
    
    if sample_size:
        df = df.head(sample_size)
    
    # Criar grafo
    B = nx.Graph()
    
    # Adicionar nós
    socios = [(f"S_{s}", {'type': 'socio'}) for s in df['nome_socio'].unique()]
    empresas = [(f"E_{e}", {'type': 'empresa'}) for e in df['cnpj_basico'].unique()]
    
    B.add_nodes_from(socios)
    B.add_nodes_from(empresas)
    
    # Adicionar arestas
    edges = [(f"S_{row['nome_socio']}", f"E_{row['cnpj_basico']}") 
             for _, row in df.iterrows()]
    B.add_edges_from(edges)
    
    print(f"✅ Grafo bipartido criado:")
    print(f"   Nós: {B.number_of_nodes():,}")
    print(f"   Arestas: {B.number_of_edges():,}")
    
    return B


def project_socios_network(B):
    """
    Cria projeção de rede de sócios a partir do grafo bipartido
    
    Args:
        B (nx.Graph): Grafo bipartido
        
    Returns:
        nx.Graph: Rede de sócios projetada
    """
    # Identificar nós por tipo
    socios = {n for n, d in B.nodes(data=True) if d.get('type') == 'socio'}
    empresas = {n for n, d in B.nodes(data=True) if d.get('type') == 'empresa'}
    
    # Criar rede de sócios
    G = nx.Graph()
    G.add_nodes_from(socios)
    
    # Conectar sócios que compartilham empresas
    for empresa in empresas:
        socios_empresa = list(B.neighbors(empresa))
        
        for i, s1 in enumerate(socios_empresa):
            for s2 in socios_empresa[i+1:]:
                if G.has_edge(s1, s2):
                    G[s1][s2]['weight'] += 1
                else:
                    G.add_edge(s1, s2, weight=1)
    
    print(f"✅ Rede de sócios projetada:")
    print(f"   Nós: {G.number_of_nodes():,}")
    print(f"   Arestas: {G.number_of_edges():,}")
    
    return G


def compute_network_metrics(G):
    """
    Calcula métricas básicas da rede
    
    Args:
        G (nx.Graph): Grafo
        
    Returns:
        dict: Dicionário com métricas
    """
    metrics = {}
    
    # Métricas básicas
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    metrics['num_components'] = nx.number_connected_components(G)
    
    # Componente maior
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        metrics['largest_component_size'] = len(largest_cc)
        metrics['largest_component_pct'] = len(largest_cc) / G.number_of_nodes()
    else:
        metrics['largest_component_size'] = 0
        metrics['largest_component_pct'] = 0
    
    # Grau médio
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['median_degree'] = np.median(degrees)
        metrics['max_degree'] = np.max(degrees)
    else:
        metrics['avg_degree'] = 0
        metrics['median_degree'] = 0
        metrics['max_degree'] = 0
    
    return metrics


def get_top_nodes(G, n=10, metric='degree'):
    """
    Retorna top N nós por métrica de centralidade
    
    Args:
        G (nx.Graph): Grafo
        n (int): Número de nós a retornar
        metric (str): Métrica ('degree', 'betweenness', 'closeness', 'eigenvector')
        
    Returns:
        list: Lista de tuplas (nó, valor)
    """
    if metric == 'degree':
        centrality = dict(G.degree())
    elif metric == 'degree_centrality':
        centrality = nx.degree_centrality(G)
    elif metric == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif metric == 'closeness':
        # Apenas para componente conectado
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        centrality = nx.closeness_centrality(G_sub)
    elif metric == 'eigenvector':
        # Apenas para componente conectado
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        centrality = nx.eigenvector_centrality(G_sub, max_iter=1000)
    else:
        raise ValueError(f"Métrica desconhecida: {metric}")
    
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]
    return top


def detect_communities(G, method='greedy'):
    """
    Detecta comunidades no grafo
    
    Args:
        G (nx.Graph): Grafo
        method (str): Método de detecção ('greedy', 'label_propagation')
        
    Returns:
        list: Lista de comunidades (cada uma é um conjunto de nós)
    """
    # Usar apenas o maior componente conectado
    if G.number_of_nodes() == 0:
        return []
    
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc).copy()
    
    if method == 'greedy':
        communities = community.greedy_modularity_communities(G_sub)
    elif method == 'label_propagation':
        communities = community.label_propagation_communities(G_sub)
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    return list(communities)


def create_centrality_dataframe(G, prefix='S_'):
    """
    Cria DataFrame com métricas de centralidade dos nós
    
    Args:
        G (nx.Graph): Grafo
        prefix (str): Prefixo dos nós a remover
        
    Returns:
        pd.DataFrame: DataFrame com centralidades
    """
    # Calcular centralidades
    degree_cent = nx.degree_centrality(G)
    
    # Componente maior para outras métricas
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)
    
    betweenness_cent = nx.betweenness_centrality(G_sub)
    closeness_cent = nx.closeness_centrality(G_sub)
    
    # Criar DataFrame
    data = []
    for node in G.nodes():
        row = {
            'node': node.replace(prefix, ''),
            'degree': G.degree(node),
            'degree_centrality': degree_cent[node],
            'betweenness_centrality': betweenness_cent.get(node, 0),
            'closeness_centrality': closeness_cent.get(node, 0)
        }
        data.append(row)
    
    df = pd.DataFrame(data).sort_values('degree', ascending=False)
    return df


def visualize_network_sample(G, n_nodes=50, figsize=(16, 12), save_path=None):
    """
    Visualiza amostra da rede (nós mais conectados)
    
    Args:
        G (nx.Graph): Grafo
        n_nodes (int): Número de nós a visualizar
        figsize (tuple): Tamanho da figura
        save_path (Path): Caminho para salvar a figura (None para não salvar)
        
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    # Selecionar top nodes por grau
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()
    
    top_nodes = sorted(G_largest.degree(), key=lambda x: x[1], reverse=True)[:n_nodes]
    top_nodes_set = {node for node, degree in top_nodes}
    G_viz = G_largest.subgraph(top_nodes_set).copy()
    
    # Layout
    pos = nx.spring_layout(G_viz, k=0.5, iterations=50, seed=42)
    
    # Tamanho dos nós baseado no grau
    node_sizes = [G_viz.degree(node) * 50 for node in G_viz.nodes()]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        G_viz,
        pos=pos,
        with_labels=False,
        node_size=node_sizes,
        node_color='lightblue',
        edge_color='gray',
        alpha=0.7,
        width=0.5,
        ax=ax
    )
    ax.set_title(f'Rede de Relacionamentos (Top {n_nodes} mais conectados)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva: {save_path}")
    
    return fig


def plot_degree_distribution(G, figsize=(12, 6), save_path=None):
    """
    Plota distribuição de graus da rede
    
    Args:
        G (nx.Graph): Grafo
        figsize (tuple): Tamanho da figura
        save_path (Path): Caminho para salvar a figura
        
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    degrees = [d for n, d in G.degree()]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(degrees, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title('Distribuição de Graus', fontsize=14, fontweight='bold')
    ax.set_xlabel('Grau (número de conexões)', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.axvline(np.mean(degrees), color='red', linestyle='--', linewidth=2, 
               label=f'Média: {np.mean(degrees):.2f}')
    ax.axvline(np.median(degrees), color='green', linestyle='--', linewidth=2, 
               label=f'Mediana: {np.median(degrees):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva: {save_path}")
    
    return fig


def print_network_summary(G):
    """
    Imprime resumo das métricas da rede
    
    Args:
        G (nx.Graph): Grafo
    """
    metrics = compute_network_metrics(G)
    
    print("=" * 60)
    print("RESUMO DA REDE")
    print("=" * 60)
    print(f"Nós: {metrics['num_nodes']:,}")
    print(f"Arestas: {metrics['num_edges']:,}")
    print(f"Densidade: {metrics['density']:.6f}")
    print(f"Componentes conectados: {metrics['num_components']:,}")
    print(f"Maior componente: {metrics['largest_component_size']:,} nós "
          f"({metrics['largest_component_pct']:.1%})")
    print(f"Grau médio: {metrics['avg_degree']:.2f}")
    print(f"Grau mediano: {metrics['median_degree']:.1f}")
    print(f"Grau máximo: {metrics['max_degree']:,}")
