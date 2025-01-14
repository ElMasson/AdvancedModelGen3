import streamlit as st
from graphviz import Digraph
from typing import Dict, List
import json
import os
import streamlit.components.v1 as components


##        mermaid.initialize({{ startOnLoad: true , theme: 'default', version: '9.0.0' }});

# Function to display Mermaid diagrams in Streamlit
def display_mermaid_diagram(mermaid_code: str):
    """Renders the Mermaid diagram, handling errors if needed."""
    if not mermaid_code.strip():
        st.error("Aucun code Mermaid valide à afficher.")
        return

    html_code = f"""
    <div class="mermaid">
        {mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        document.addEventListener("DOMContentLoaded", () => {{
            mermaid.initialize({{ startOnLoad: true }});
            mermaid.run();
        }});
    </script>
    """

    components.html(html_code, height=1000, scrolling=True)


# Safe loading and saving of JSON files
def safe_load_json(file_path: str, default_data: dict) -> dict:
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            st.warning(f"Fichier {file_path} non trouvé. Utilisation des données par défaut.")
            return default_data
    except json.JSONDecodeError:
        st.error(f"Erreur de décodage du fichier {file_path}. Utilisation des données par défaut.")
        return default_data
    except Exception as e:
        st.error(f"Erreur lors du chargement de {file_path}: {str(e)}")
        return default_data

def safe_save_json(file_path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        st.success(f"Données sauvegardées dans {file_path}")
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde dans {file_path}: {str(e)}")

# Function to create ER diagrams
def create_table_node(dot: Digraph, table_name: str, columns: List[str], source: str = ""):
    table_html = f'''<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0">
            <TR><TD PORT="header" BGCOLOR="lightgrey"><B>{table_name} ({source})</B></TD></TR>
    '''
    for col in columns:
        table_html += f'<TR><TD PORT="{col}">{col}</TD></TR>'
    table_html += '</TABLE>>'
    dot.node(table_name, table_html)


# Function to create ER diagram nodes with correct source labels
def create_er_diagram_with_source(tables: Dict, relationships: List, title: str) -> Digraph:
    dot = Digraph(comment=title)
    dot.attr('node', shape='plaintext')
    dot.attr('graph', rankdir='LR')

    # Define the sources for each table
    table_sources = {
        "ARTISTES": ["duke_demo_artists", "duke_demo_events"],
        "TCC_CEDS_MUSIC": ["tcc_ceds_music_catalog.csv"],
        "OEUVRES_MUSICALES": ["duke_demo_artists"],
        "ALBUMS": ["duke_demo_artists"],
        "CONCERTS": ["duke_demo_events"],
        "RECORDING": ["duke_demo_events"],
        "ROYALTIES": ["duke_demo_events"],
        "GENRES_MUSICAUX": ["duke_demo_artists"],
        "COLLABORATIONS": ["duke_demo_artists"],
        "COLLABORATION_OEUVRE": ["duke_demo_artists"],
        "COMPOSITEURS": ["duke_demo_artists"],
        "INSTRUMENTS": ["duke_demo_artists"],
        "LANGUAGE": ["duke_demo_artists"],
        "LIEU": ["duke_demo_artists"],
        "MANAGEMENTS": ["duke_demo_artists"],
        "PRODUCTEUR": ["duke_demo_artists"],
        "REGION": ["duke_demo_artists"],
        "LABELS": ["duke_demo_artists"],
        "CONTRAT": ["duke_demo_events"]
    }

    # Create the tables with their correct source
    for table_name, columns in tables.items():
        # Determine the correct source based on table name
        source = ", ".join(table_sources.get(table_name, ["multisource"]))
        create_table_node(dot, table_name, columns, source)

    # Add the relationships between tables
    for rel in relationships:
        from_table = rel['from']['table']
        to_table = rel['to']['table']
        from_col = rel['from']['column'] if isinstance(rel['from']['column'], str) else ", ".join(rel['from']['column'])
        to_col = rel['to']['column']
        cardinality = f"{rel['from']['cardinality']}-{rel['to']['cardinality']}"
        functional = rel.get('functional_label', '')

        label = f"{functional}\n({from_col} -> {to_col})\n{cardinality}\n({rel['probability']}%)"

        dot.edge(
            f"{from_table}:header",
            f"{to_table}:header",
            label=label,
            style=get_relationship_style(rel['probability'])
        )
    return dot




def get_relationship_style(probability: int) -> str:
    if probability == 100:
        return 'solid'
    elif probability >= 75:
        return 'dashed'
    elif probability >= 50:
        return 'dotted'
    else:
        return 'dashed,bold'

# Function to create Mermaid diagrams for ontologies
def create_ontology_diagram(ontology: dict) -> str:
    """Generates the Mermaid class diagram for the ontology."""
    mermaid_code = "classDiagram\n"

    for concept, details in ontology.items():
        # Add the class (concept)
        mermaid_code += f"class {concept} {{\n"

        # Ajouter une vérification pour `description`
        description = details.get('description', 'No description available')
        mermaid_code += f"  // {description}\n"

        for prop in details.get('properties', []):
            mermaid_code += f"  +{prop}\n"
        mermaid_code += "}\n"

    # Add relationships between the concepts
    for concept, details in ontology.items():
        for rel in details.get('relationships', []):
            target = rel.split()[-1]
            label = ' '.join(rel.split()[:-1])
            mermaid_code += f"{concept} --> {target} : {label}\n"

    return mermaid_code


# Function to edit the tables in the data model

def edit_tables(tables: Dict) -> Dict:
    st.subheader("Éditer les tables")
    edited_tables = tables.copy()

    for table_name, columns in edited_tables.items():
        with st.expander(f"Table: {table_name}"):
            new_columns = st.text_area(f"Colonnes pour {table_name}", value='\n'.join(columns)).split('\n')
            edited_tables[table_name] = [col.strip() for col in new_columns if col.strip()]

    if st.button("Sauvegarder les modifications des tables"):
        safe_save_json(os.path.join("data", "tables.json"), edited_tables)
        st.success("Tables mises à jour avec succès!")

    return edited_tables


# Function to edit ontology concepts, properties, and relationships
def edit_ontology(ontology: dict) -> dict:
    st.subheader("Éditer l'ontologie")
    edited_ontology = ontology.copy()

    for concept, details in edited_ontology.items():
        with st.expander(f"Concept: {concept}"):
            new_description = st.text_area(f"Description pour {concept}", value=details['description'])
            edited_ontology[concept]['description'] = new_description

            st.markdown("**Propriétés:**")
            new_properties = st.text_area(f"Propriétés pour {concept}", value='\n'.join(details['properties'])).split(
                '\n')
            edited_ontology[concept]['properties'] = [prop.strip() for prop in new_properties if prop.strip()]

            st.markdown("**Relations:**")
            new_relationships = st.text_area(f"Relations pour {concept}",
                                             value='\n'.join(details['relationships'])).split('\n')
            edited_ontology[concept]['relationships'] = [rel.strip() for rel in new_relationships if rel.strip()]

    if st.button("Sauvegarder les modifications de l'ontologie"):
        safe_save_json(os.path.join("data", "ontology.json"), edited_ontology)
        st.success("Ontologie mise à jour avec succès!")

    return edited_ontology



# Function to edit relationships in the data model
def edit_relationships(relationships: List) -> List:
    st.subheader("Éditer les relations")
    edited_relationships = relationships.copy()

    for i, rel in enumerate(edited_relationships):
        with st.expander(f"Relation {i + 1}: {rel['from']} → {rel['to']}"):
            col1, col2 = st.columns(2)
            with col1:
                rel['from'] = st.text_input("De", rel['from'], key=f"from_{i}")
                rel['to'] = st.text_input("Vers", rel['to'], key=f"to_{i}")
            with col2:
                rel['label'] = st.text_input("Label", rel['label'], key=f"label_{i}")
                rel['probability'] = st.slider("Probabilité", 0, 100, rel['probability'], key=f"prob_{i}")

            if st.button("Supprimer cette relation", key=f"del_{i}"):
                edited_relationships.pop(i)
                st.experimental_rerun()

    with st.expander("Ajouter une nouvelle relation"):
        new_from = st.text_input("De", "")
        new_to = st.text_input("Vers", "")
        new_label = st.text_input("Label", "")
        new_prob = st.slider("Probabilité", 0, 100, 50)
        if st.button("Ajouter") and new_from and new_to:
            edited_relationships.append({
                'from': new_from,
                'to': new_to,
                'label': new_label,
                'probability': new_prob
            })
            st.experimental_rerun()

    if st.button("Sauvegarder les modifications des relations"):
        safe_save_json(os.path.join("data", "relationships.json"), edited_relationships)
        st.success("Relations mises à jour avec succès!")

    return edited_relationships

# Function to display ontology details with source of each property
def show_ontology_details(ontology: dict, editable: bool = False):
    """Displays ontology details with the option to edit."""
    for concept, details in ontology.items():
        with st.expander(f"📚 {concept}"):
            # Utiliser get pour éviter l'erreur si 'description' est absent
            description = details.get('description', 'Aucune description disponible.')
            st.markdown(f"**Description:** {description}")

            st.markdown("**🔑 Propriétés:**")
            for prop in details.get('properties', []):
                st.markdown(f"- {prop}")

            st.markdown("**🔗 Relations:**")
            for rel in details.get('relationships', []):
                st.markdown(f"- {rel}")

            if 'source' in details:
                st.markdown("**📂 Sources:**")
                for source in details['source']:
                    st.markdown(f"- {source}")

            # Allow editing of the ontology if editable is True
            if editable:
                st.text_area("Modifier les propriétés et relations", value=json.dumps(details, indent=2), height=300)



# Function to create tabs for the ontology views
def create_ontology_tabs(specific_ontology: dict, editable: bool = False) -> dict:
    """Creates tabs for visualizing, detailing, and editing a specific ontology."""
    viz_tab, details_tab, json_tab = st.tabs([
        "Visualisation", "Détails", "JSON"
    ])

    # Visualization (Mermaid Diagram)
    with viz_tab:
        mermaid_code = create_ontology_diagram(specific_ontology)
        display_mermaid_diagram(mermaid_code)
        st.markdown("### Légende")
        st.markdown("""
        - 📚 Concept de l'ontologie
        - 🔑 Propriété avec la source
        - 🔗 Relation entre concepts
        """)

    # Details view (editable or not depending on the checkbox)
    with details_tab:
        show_ontology_details(specific_ontology, editable=editable)

    # JSON View with editing option
    with json_tab:
        if editable:
            edited_json = st.text_area("Éditer JSON", value=json.dumps(specific_ontology, indent=2), height=400)
            if st.button("Sauvegarder l'ontologie en JSON"):
                try:
                    # Convert JSON data to dictionary
                    new_specific_ontology = json.loads(edited_json)
                    return new_specific_ontology
                except json.JSONDecodeError:
                    st.error("JSON invalide")
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde : {str(e)}")
        else:
            st.json(specific_ontology)

    return specific_ontology


# Function to display interactive Mermaid diagrams in Streamlit
def display_interactive_mermaid_diagram(mermaid_code: str):
    """Renders the Mermaid diagram with zoomable and pannable functionality."""
    if not mermaid_code.strip():
        st.error("Aucun code Mermaid valide à afficher.")
        return

    # Adjusting the zoom and pan settings with svgPanZoom
    html_code = f"""
    <div id="mermaid-container" style="text-align:center;">
        <div class="mermaid" id="mermaid-diagram">
            {mermaid_code}
        </div>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        import svgPanZoom from 'https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js';

        // Initialize Mermaid
        mermaid.initialize({{ startOnLoad: true }});

        // Wait until the diagram is rendered before applying zoom and pan
        setTimeout(() => {{
            const svgElement = document.querySelector('svg');
            if (svgElement) {{
                svgPanZoom(svgElement, {{
                    zoomEnabled: true,
                    controlIconsEnabled: true,
                    fit: true,
                    center: true,
                    minZoom: 0.5,
                    maxZoom: 10
                }});
            }} else {{
                console.error("SVG Element not found for zooming.");
            }}
        }}, 2000);
    </script>
    """
    # Render the diagram in Streamlit with a larger height for better visualization
    components.html(html_code, height=800, scrolling=True)

# Main function for Streamlit app with zoomable and pannable Mermaid diagram
def main():
    st.title("Modélisation des données musicales")

    os.makedirs("data", exist_ok=True)

    # Loading the data
    tables = safe_load_json(os.path.join("data", "tables.json"), {})
    relationships = safe_load_json(os.path.join("data", "relationships.json"), {})
    ontology = safe_load_json(os.path.join("data", "ontology.json"), {})

    # Utiliser la barre latérale pour le menu de sélection
    page = st.sidebar.radio(
        "Choisissez une section",
        ("MCD Global", "Ontologie Globale", "MCD - Duke Demo Artists",
         "Ontologie - Duke Demo Artists", "MCD - Duke Demo Events",
         "Ontologie - Duke Demo Events", "Ontologie - Fichier CSV")
    )

    # Affichage de la page sélectionnée
    if page == "MCD Global":
        st.header("Modèle conceptuel de données global")
        if st.checkbox("Éditer les tables", key="edit_global_tables"):
            tables['global'] = edit_tables(tables['global'])
        if st.checkbox("Éditer les relations", key="edit_global_relationships"):
            relationships['global'] = edit_relationships(relationships['global'])
        dot = create_er_diagram_with_source(tables['global'], relationships['global'], "Modèle Global")
        st.graphviz_chart(dot)

    elif page == "Ontologie Globale":
        st.header("Ontologie globale")
        editable = st.checkbox("Éditer l'ontologie globale", key="edit_global_ontology")
        edited_global_ontology = create_ontology_tabs(ontology.get('global', {}), editable)
        if edited_global_ontology != ontology.get('global'):
            ontology['global'] = edited_global_ontology
            safe_save_json(os.path.join("data", "ontology.json"), ontology)
            st.success("Ontologie globale mise à jour avec succès!")

    elif page == "MCD - Duke Demo Artists":
        st.header("Modèle conceptuel - Duke Demo Artists")
        if st.checkbox("Éditer les tables", key="edit_duke_artists_tables"):
            tables['duke_demo_artists'] = edit_tables(tables['duke_demo_artists'])
        if st.checkbox("Éditer les relations", key="edit_duke_artists_relationships"):
            relationships['duke_demo_artists'] = edit_relationships(relationships['duke_demo_artists'])
        dot = create_er_diagram_with_source(tables.get('duke_demo_artists', {}),
                                            relationships.get('duke_demo_artists', []), "Duke Demo Artists")
        st.graphviz_chart(dot)

    elif page == "Ontologie - Duke Demo Artists":
        st.header("Ontologie - Duke Demo Artists")
        editable = st.checkbox("Éditer l'ontologie Duke Demo Artists", key="edit_duke_artists_ontology")
        edited_duke_artists_ontology = create_ontology_tabs(ontology['schemas'].get('duke_demo_artists', {}), editable)
        if edited_duke_artists_ontology != ontology['schemas'].get('duke_demo_artists'):
            ontology['schemas']['duke_demo_artists'] = edited_duke_artists_ontology
            safe_save_json(os.path.join("data", "ontology.json"), ontology)
            st.success("Ontologie Duke Demo Artists mise à jour avec succès!")

    elif page == "MCD - Duke Demo Events":
        st.header("Modèle conceptuel - Duke Demo Events")
        if st.checkbox("Éditer les tables", key="edit_duke_events_tables"):
            tables['duke_demo_events'] = edit_tables(tables['duke_demo_events'])
        if st.checkbox("Éditer les relations", key="edit_duke_events_relationships"):
            relationships['duke_demo_events'] = edit_relationships(relationships['duke_demo_events'])
        dot = create_er_diagram_with_source(tables.get('duke_demo_events', {}),
                                            relationships.get('duke_demo_events', []), "Duke Demo Events")
        st.graphviz_chart(dot)

    elif page == "Ontologie - Duke Demo Events":
        st.header("Ontologie - Duke Demo Events")
        editable = st.checkbox("Éditer l'ontologie Duke Demo Events", key="edit_duke_events_ontology")
        edited_duke_events_ontology = create_ontology_tabs(ontology['schemas'].get('duke_demo_events', {}), editable)
        if edited_duke_events_ontology != ontology['schemas'].get('duke_demo_events'):
            ontology['schemas']['duke_demo_events'] = edited_duke_events_ontology
            safe_save_json(os.path.join("data", "ontology.json"), ontology)
            st.success("Ontologie Duke Demo Events mise à jour avec succès!")

    elif page == "Ontologie - Fichier CSV":
        st.header("Ontologie - Fichier CSV")
        editable = st.checkbox("Éditer l'ontologie du fichier CSV", key="edit_csv_ontology")
        edited_csv_ontology = create_ontology_tabs(ontology['schemas'].get('tcc_ceds_music', {}), editable)
        if edited_csv_ontology != ontology['schemas'].get('tcc_ceds_music'):
            ontology['schemas']['tcc_ceds_music'] = edited_csv_ontology
            safe_save_json(os.path.join("data", "ontology.json"), ontology)
            st.success("Ontologie du fichier CSV mise à jour avec succès!")

if __name__ == "__main__":
    main()