# visualization/knowledge_graph_view.py
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphVisualizer:
    def __init__(self):
        self.default_layout_settings = {
            'showlegend': False,
            'hovermode': 'closest',
            'margin': dict(b=20, l=5, r=5, t=40),
            'height': 800
        }

    def create_visualization(self, nodes: List[Dict], edges: List[Dict]) -> go.Figure:
        """Crée une visualisation interactive du graphe de connaissances."""
        # Création du graphe
        G = self._create_network(nodes, edges)

        # Calcul du layout
        pos = nx.spring_layout(G)

        # Création des traces
        edge_trace = self._create_edge_trace(G, pos)
        node_trace = self._create_node_trace(G, pos)

        # Création de la figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=self._create_layout()
        )

        return fig

    def _create_network(self, nodes: List[Dict], edges: List[Dict]) -> nx.Graph:
        """Crée un graphe NetworkX à partir des nœuds et arêtes."""
        G = nx.Graph()

        # Ajout des nœuds
        for node in nodes:
            G.add_node(node['id'], **node.get('data', {}))

        # Ajout des arêtes
        for edge in edges:
            G.add_edge(
                edge['source'],
                edge['target'],
                **edge.get('data', {})
            )

        return G

    def _create_edge_trace(self, G: nx.Graph, pos: Dict) -> go.Scatter:
        """Crée la trace pour les arêtes du graphe."""
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2].get('type', 'relation'))

        return go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )

    def _create_node_trace(self, G: nx.Graph, pos: Dict) -> go.Scatter:
        """Crée la trace pour les nœuds du graphe."""
        node_x = []
        node_y = []
        node_text = []
        node_color = []

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)

            # Création du texte au survol
            hover_text = f"""
            Table: {node[0]}
            Type: {node[1].get('type', 'N/A')}
            Description: {node[1].get('description', 'N/A')}
            """
            node_text.append(hover_text)

            # Couleur basée sur le nombre de connexions
            node_color.append(len(list(G.neighbors(node[0]))))

        return go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_color,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Nombre de connexions',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )

    def _create_layout(self) -> go.Layout:
        """Crée le layout de la visualisation."""
        return go.Layout(
            title='Graphe de connaissances de la base de données',
            titlefont_size=16,
            **self.default_layout_settings,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002
                )
            ]
        )

    def update_graph(self, new_nodes: List[Dict], new_edges: List[Dict]) -> go.Figure:
        """Met à jour le graphe avec de nouveaux nœuds et arêtes."""
        return self.create_visualization(new_nodes, new_edges)

    def export_to_html(self, fig: go.Figure, filename: str = "knowledge_graph.html"):
        """Exporte la visualisation en fichier HTML."""
        try:
            fig.write_html(filename)
            logger.info(f"Graph exported successfully to {filename}")
        except Exception as e:
            logger.error(f"Error exporting graph: {str(e)}")
            raise