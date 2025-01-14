# results_view.py
from visualization.data_model_component import data_model_view

class ResultsView:
    # ... autres méthodes ...

    def _render_data_model_tab(self, schema_results: Dict):
        """
        Affiche l'onglet du modèle de données.
        """
        try:
            st.header("Modèle de Données")

            # Préparation des données
            model_data = self._prepare_schema_data(schema_results)

            # Utilisation du composant
            data_model_view(
                data=model_data,
                key="data_model_view"
            )

        except Exception as e:
            logger.error(f"Error rendering data model tab: {str(e)}")
            st.error("Erreur lors de l'affichage du modèle de données")