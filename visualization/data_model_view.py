# visualization/data_model_view.py
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict
import json


class DataModelView:
    """Composant de visualisation du modèle de données utilisant Streamlit."""

    def __init__(self):
        # Code JavaScript minimal requis
        self.js_code = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
        <script>
        function createDataModelView(container, data) {
            const width = 800;
            const height = 600;

            // Création du SVG
            const svg = d3.select(container)
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("viewBox", [0, 0, width, height]);

            // Configuration force layout
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.edges).id(d => d.id))
                .force("charge", d3.forceManyBody())
                .force("center", d3.forceCenter(width / 2, height / 2));

            // Création des liens
            const link = svg.append("g")
                .selectAll("line")
                .data(data.edges)
                .join("line")
                .attr("stroke", "#999")
                .attr("stroke-opacity", 0.6);

            // Création des nœuds
            const node = svg.append("g")
                .selectAll("g")
                .data(data.nodes)
                .join("g")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Ajout des cercles pour les nœuds
            node.append("circle")
                .attr("r", 20)
                .attr("fill", d => d.type === 'table' ? "#4299e1" : "#48bb78");

            // Ajout des labels
            node.append("text")
                .text(d => d.label)
                .attr("x", 0)
                .attr("y", 30)
                .attr("text-anchor", "middle");

            // Fonctions de drag
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }

            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }

            // Mise à jour des positions
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("transform", d => `translate(${d.x},${d.y})`);
            });
        }
        </script>
        """

    def render(self, data: Dict):
        """
        Affiche la visualisation du modèle de données.

        Args:
            data: Données du modèle à visualiser
        """
        # Container pour la visualisation
        container_id = "data-model-container"

        # HTML avec le code JavaScript et le conteneur
        html_content = f"""
        {self.js_code}
        <div id="{container_id}" style="width:100%;height:600px;"></div>
        <script>
            const data = {json.dumps(data)};
            createDataModelView(document.getElementById("{container_id}"), data);
        </script>
        """

        # Affichage via Streamlit
        components.html(html_content, height=600)