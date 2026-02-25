# insight_generator.py

from typing import Dict, List, Tuple, Any
from collections import Counter


class InsightGenerator:
    """
    Responsible ONLY for converting ranked similar cases
    into structured clinical insight.
    """

    def __init__(self, case_database: Dict[str, Dict[str, Any]]):
        """
        Parameters:
        -----------
        case_database : dict
            Dictionary of {case_id: full_case_data}
        """
        self.case_database = case_database

    # ---------------------------------------------------
    # Public Method
    # ---------------------------------------------------

    def generate_insight(
        self,
        top_matches: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Generate structured insight from top matched cases.

        Returns:
        --------
        {
            "recommended_treatment": str,
            "most_common_diagnosis": str,
            "confidence_note": str
        }
        """

        diagnoses = []
        treatments = []

        for case_id, score in top_matches:
            case_data = self.case_database.get(case_id, {})

            diagnosis = case_data.get("diagnosis")
            treatment = case_data.get("treatment")

            if diagnosis:
                diagnoses.append(diagnosis)

            if treatment:
                treatments.append(treatment)

        most_common_diagnosis = self._most_common(diagnoses)
        recommended_treatment = self._most_common(treatments)

        return {
            "most_common_diagnosis": most_common_diagnosis,
            "recommended_treatment": recommended_treatment,
            "confidence_note": self._generate_confidence_note(top_matches)
        }

    # ---------------------------------------------------
    # Private Methods
    # ---------------------------------------------------

    @staticmethod
    def _most_common(items: List[str]) -> str:
        if not items:
            return "Insufficient data"

        counter = Counter(items)
        return counter.most_common(1)[0][0]

    @staticmethod
    def _generate_confidence_note(
        top_matches: List[Tuple[str, float]]
    ) -> str:
        if not top_matches:
            return "No similar historical cases found."

        avg_score = sum(score for _, score in top_matches) / len(top_matches)

        if avg_score > 0.85:
            return "High confidence based on strong similarity with historical cases."
        elif avg_score > 0.65:
            return "Moderate confidence based on similarity patterns."
        else:
            return "Low similarity confidence. Consider additional clinical evaluation."