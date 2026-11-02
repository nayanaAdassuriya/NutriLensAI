"""
Comprehensive Health and Food Safety Audit Engine for NutriLens AI.

This module evaluates nutritional metrics and ingredient lists against:
1. Sri Lankan Ministry of Health Food (Colour Coding for Sugar, Salt, and Fat)
   Regulations (Gazette No. 2128/9 and 2176/18).
2. Demographic Suitability: Pregnant mothers, infants/toddlers, and elderly.
3. Clinical NCD Restrictions: Diabetes Mellitus, Hypertension, CVD, and CKD.
4. Frequency Impact Analysis: Metabolic and systemic risks of daily vs. occasional intake.
5. Alarming Additives, Carcinogens, Endocrine Disruptors, and Synthetic Dyes.
6. Regulatory Discrepancy Checks: Global (EU EFSA) vs. Local (SL MOH / Codex) standards.
7. Tri-color Unified Safety Rating: Green (Safe), Orange (Moderate Caution), Red (Critical Risk).
"""

import re
from typing import Dict, List, Any, Tuple
from data_pipeline import extract_e_numbers


class SafetyAuditEngine:
    """
    Rule-based clinical and regulatory audit engine providing deep safety assessments.
    """
    
    # Sri Lankan MOH Traffic Light Thresholds (Solid Foods: g/100g, Sodium: mg/100g)
    # Ref: Ministry of Health Nutrition Division Food Act Regulations
    MOH_THRESHOLDS = {
        "solid": {
            "sugar": {"green": 5.0, "amber": 22.0},       # < 5g Green, 5-22g Amber, > 22g Red
            "salt_sodium": {"green": 100.0, "amber": 500.0}, # < 100mg Green, 100-500mg Amber, > 500mg Red
            "fat": {"green": 3.0, "amber": 17.5}          # < 3g Green, 3-17.5g Amber, > 17.5g Red
        },
        "beverage": {
            "sugar": {"green": 2.0, "amber": 12.0},       # < 2g Green, 2-12g Amber, > 12g Red
            "salt_sodium": {"green": 50.0, "amber": 250.0},
            "fat": {"green": 1.5, "amber": 8.0}
        }
    }
    
    # Known toxicological and regulatory flag lists
    EU_BANNED_OR_RESTRICTED = {
        "E171": {
            "name": "Titanium Dioxide",
            "reason": "Banned across the European Union (EU Reg 2022/63) due to genotoxicity and DNA strand-break concerns.",
            "category": "Suspected Genotoxin / Whitening Agent"
        },
        "E924": {
            "name": "Potassium Bromate",
            "reason": "Banned in EU, UK, Canada, and Sri Lanka (SLS standards) as an IARC Group 2B human carcinogen.",
            "category": "Flour Bleaching Agent / Carcinogen"
        },
        "E924A": {
            "name": "Potassium Bromate",
            "reason": "Banned in EU, UK, and Sri Lanka as an IARC Group 2B human carcinogen.",
            "category": "Carcinogen"
        },
        "E952": {
            "name": "Sodium Cyclamate",
            "reason": "Banned in the US and strictly regulated in EU due to bladder carcinogenesis in animal models.",
            "category": "Artificial Sweetener"
        },
        "E284": {
            "name": "Boric Acid",
            "reason": "Prohibited in EU and Sri Lanka due to reproductive toxicity and organ accumulation.",
            "category": "Banned Preservative"
        }
    }
    
    SOUTHAMPTON_SIX_DYES = {
        "E102": "Tartrazine (Yellow #5)",
        "E110": "Sunset Yellow FCF (Yellow #6)",
        "E122": "Carmoisine (Azorubine)",
        "E124": "Ponceau 4R",
        "E129": "Allura Red AC (Red #40)",
        "E104": "Quinoline Yellow"
    }
    
    ALARMING_ADDITIVES = {
        "E320": {
            "name": "Butylated Hydroxyanisole (BHA)",
            "hazard": "Endocrine disruptor and reasonably anticipated human carcinogen (NTP).",
            "severity": "HIGH"
        },
        "E321": {
            "name": "Butylated Hydroxytoluene (BHT)",
            "hazard": "Suspected endocrine toxicant linked to thyroid and kidney enlargement in chronic assays.",
            "severity": "HIGH"
        },
        "E211": {
            "name": "Sodium Benzoate",
            "hazard": "Can react with Ascorbic Acid (Vitamin C / E300) in acidic conditions to synthesize carcinogenic Benzene.",
            "severity": "MODERATE"
        },
        "E250": {
            "name": "Sodium Nitrite",
            "hazard": "Forms carcinogenic nitrosamines under gastric acidity during high-temperature digestion.",
            "severity": "HIGH"
        },
        "E251": {
            "name": "Sodium Nitrate",
            "hazard": "Precursor to gastric nitrosamines; linked to colorectal cancer risks.",
            "severity": "HIGH"
        },
        "E951": {
            "name": "Aspartame",
            "hazard": "Classified by WHO IARC as Group 2B (Possibly carcinogenic to humans). Not suitable for phenylketonuria.",
            "severity": "MODERATE"
        },
        "E950": {
            "name": "Acesulfame Potassium",
            "hazard": "Contains methylene chloride breakdown residues; alters gut microbiota diversity.",
            "severity": "MODERATE"
        },
        "E621": {
            "name": "Monosodium Glutamate (MSG)",
            "hazard": "Excitotoxic flavor enhancer triggering headaches and transient hypertension in sensitive individuals.",
            "severity": "LOW"
        }
    }
    
    PHOSPHATE_ADDITIVES = ["E338", "E339", "E340", "E341", "E343", "E450", "E451", "E452"]

    def __init__(self):
        pass

    def evaluate_moh_traffic_lights(self, sugars_g: float, sodium_mg: float, fat_g: float, is_beverage: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Calculates the Sri Lankan Ministry of Health 3-color traffic light rating
        for Sugar, Salt/Sodium, and Total Fat.
        """
        matrix_type = "beverage" if is_beverage else "solid"
        thresholds = self.MOH_THRESHOLDS[matrix_type]
        
        # Sugar evaluation
        if sugars_g < thresholds["sugar"]["green"]:
            sugar_status = {"color": "Green", "level": "Low Sugar", "threshold": f"< {thresholds['sugar']['green']}g"}
        elif sugars_g <= thresholds["sugar"]["amber"]:
            sugar_status = {"color": "Amber", "level": "Medium Sugar", "threshold": f"{thresholds['sugar']['green']}-{thresholds['sugar']['amber']}g"}
        else:
            sugar_status = {"color": "Red", "level": "High Sugar", "threshold": f"> {thresholds['sugar']['amber']}g"}
            
        # Sodium evaluation
        if sodium_mg < thresholds["salt_sodium"]["green"]:
            sodium_status = {"color": "Green", "level": "Low Sodium", "threshold": f"< {thresholds['salt_sodium']['green']}mg"}
        elif sodium_mg <= thresholds["salt_sodium"]["amber"]:
            sodium_status = {"color": "Amber", "level": "Medium Sodium", "threshold": f"{thresholds['salt_sodium']['green']}-{thresholds['salt_sodium']['amber']}mg"}
        else:
            sodium_status = {"color": "Red", "level": "High Sodium", "threshold": f"> {thresholds['salt_sodium']['amber']}mg"}
            
        # Fat evaluation
        if fat_g < thresholds["fat"]["green"]:
            fat_status = {"color": "Green", "level": "Low Fat", "threshold": f"< {thresholds['fat']['green']}g"}
        elif fat_g <= thresholds["fat"]["amber"]:
            fat_status = {"color": "Amber", "level": "Medium Fat", "threshold": f"{thresholds['fat']['green']}-{thresholds['fat']['amber']}g"}
        else:
            fat_status = {"color": "Red", "level": "High Fat", "threshold": f"> {thresholds['fat']['amber']}g"}
            
        return {
            "sugar": sugar_status,
            "sodium": sodium_status,
            "fat": fat_status
        }

    def audit_alarming_ingredients(self, ingredient_text: str) -> Dict[str, Any]:
        """
        Scans ingredient text and detected E-numbers for toxic chemicals,
        carcinogens, synthetic dyes, and benzene precursors.
        """
        detected_e_numbers = extract_e_numbers(ingredient_text)
        text_lower = ingredient_text.lower()
        
        flags = []
        regulatory_discrepancies = []
        carcinogens_and_toxins = []
        southampton_dyes = []
        
        # Check EU banned/restricted additives
        for e_num, info in self.EU_BANNED_OR_RESTRICTED.items():
            if e_num in detected_e_numbers or info["name"].lower() in text_lower:
                regulatory_discrepancies.append({
                    "code": e_num,
                    "name": info["name"],
                    "category": info["category"],
                    "reason": info["reason"]
                })
                
        # Check Southampton 6 ADHD dyes
        for e_num, dye_name in self.SOUTHAMPTON_SIX_DYES.items():
            if e_num in detected_e_numbers or dye_name.lower().split(" ")[0] in text_lower:
                southampton_dyes.append({
                    "code": e_num,
                    "name": dye_name,
                    "warning": "Mandated by EU Reg 1333/2008 to state: 'May have an adverse effect on activity and attention in children'."
                })
                
        # Check Alarming additives
        for e_num, info in self.ALARMING_ADDITIVES.items():
            if e_num in detected_e_numbers or info["name"].lower() in text_lower:
                carcinogens_and_toxins.append({
                    "code": e_num,
                    "name": info["name"],
                    "hazard": info["hazard"],
                    "severity": info["severity"]
                })
                
        # Benzene precursor reaction check: Sodium Benzoate (E211) + Ascorbic Acid (E300 / Vitamin C)
        has_benzoate = ("e211" in [e.lower() for e in detected_e_numbers]) or ("benzoate" in text_lower)
        has_ascorbic = ("e300" in [e.lower() for e in detected_e_numbers]) or ("ascorbic" in text_lower) or ("vitamin c" in text_lower)
        benzene_risk = has_benzoate and has_ascorbic
        if benzene_risk:
            carcinogens_and_toxins.append({
                "code": "E211 + E300",
                "name": "Sodium Benzoate & Ascorbic Acid Synergy",
                "hazard": "CRITICAL: Simultaneous presence of benzoate and ascorbic acid in acidic aqueous conditions catalyzes toxic Benzene formation (IARC Group 1 Carcinogen).",
                "severity": "CRITICAL"
            })
            
        # Trans fats check
        trans_fat_terms = ["hydrogenated", "partially hydrogenated", "vanaspati", "shortening", "margarine"]
        has_trans_fat = any(term in text_lower for term in trans_fat_terms)
        if has_trans_fat:
            carcinogens_and_toxins.append({
                "code": "TFA",
                "name": "Industrial Trans-Fatty Acids / Hydrogenated Oils",
                "hazard": "Significantly elevates LDL cholesterol, reduces HDL, and promotes systemic vascular endothelitis.",
                "severity": "HIGH"
            })
            
        return {
            "detected_e_numbers": detected_e_numbers,
            "regulatory_discrepancies": regulatory_discrepancies,
            "carcinogens_and_toxins": carcinogens_and_toxins,
            "southampton_dyes": southampton_dyes,
            "benzene_risk": benzene_risk,
            "trans_fat_risk": has_trans_fat
        }

    def audit_demographic_suitability(
        self,
        ingredient_text: str,
        sugars_g: float,
        sodium_mg: float,
        saturated_fat_g: float,
        detected_e_numbers: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Assesses clinical safety for vulnerable demographic cohorts:
        1. Pregnant & Nursing Mothers
        2. Babies (< 1 year)
        3. Infants & Toddlers (1 - 5 years)
        4. Very Old / Geriatric Individuals
        """
        text_lower = ingredient_text.lower()
        
        # 1. Pregnant & Nursing Mothers
        preg_warnings = []
        if ("caffeine" in text_lower) or ("coffee extract" in text_lower):
            preg_warnings.append("Caffeine crosses placental barrier; impairs fetal microcirculation and elevates low birth weight risk.")
        if any(e in detected_e_numbers for e in ["E954", "E952", "E951"]):
            preg_warnings.append("Artificial intense sweeteners (Saccharin, Cyclamate, Aspartame) penetrate placenta and amnion.")
        if any(e in detected_e_numbers for e in ["E249", "E250", "E251", "E252"]):
            preg_warnings.append("Inorganic nitrates/nitrites present: risk of maternal methemoglobinemia and fetal hypoxia.")
        if sodium_mg > 500.0:
            preg_warnings.append(f"Elevated sodium load ({sodium_mg:.0f}mg/100g) exacerbates gestational hypertension and pre-eclampsia risks.")
        if saturated_fat_g > 6.0:
            preg_warnings.append("High saturated lipid content accelerates maternal insulin resistance and gestational diabetes.")
            
        preg_status = "SAFE" if len(preg_warnings) == 0 else ("MODERATE CAUTION" if len(preg_warnings) <= 2 else "RESTRICTED / HIGH RISK")
        preg_impact = "Formulation compliant with perinatal nutritional safety." if not preg_warnings else "May elevate gestational blood pressure, placenta crossing xenobiotics, or maternal metabolic stress."
        
        # 2. Babies (< 1 Year)
        baby_warnings = []
        if sodium_mg > 100.0:
            baby_warnings.append(f"Sodium content ({sodium_mg:.0f}mg/100g) severely overloads immature infant glomerular nephrons (safe ceiling < 100mg).")
        if sugars_g > 2.0:
            baby_warnings.append(f"Added sugars ({sugars_g:.1f}g/100g) violate WHO pediatric guidelines: zero added sugars mandated under 12 months.")
        if "honey" in text_lower:
            baby_warnings.append("CRITICAL: Contains honey; presents fatal infant botulism hazard from Clostridium botulinum spores.")
        if any(e in detected_e_numbers for e in ["E621", "E627", "E631"]):
            baby_warnings.append("Glutamate / ribonucleotide flavor enhancers contraindicated for developing infant central nervous system.")
        if len(detected_e_numbers) > 0:
            baby_warnings.append("Synthetic additives and chemical stabilizers disrupt infant mucosal gut colonization.")
            
        baby_status = "SUITABLE" if len(baby_warnings) == 0 else ("CAUTION" if len(baby_warnings) == 1 else "UNSUITABLE / HIGH RISK")
        baby_impact = "Gentle, non-straining formula safe for infant renal and digestive clearance." if not baby_warnings else "Substantial osmotic renal load and metabolic disruption for infant physiology under 12 months."

        # 3. Infants & Toddlers (1 - 5 Years)
        toddler_warnings = []
        if sugars_g > 5.0:
            toddler_warnings.append(f"Sugar concentration ({sugars_g:.1f}g/100g) fosters pediatric dental caries, insulin resistance, and taste desensitization.")
        if sodium_mg > 300.0:
            toddler_warnings.append(f"Sodium ({sodium_mg:.0f}mg/100g) exceeds daily toddler dietary allowance benchmarks.")
        if any(e in detected_e_numbers for e in list(self.SOUTHAMPTON_SIX_DYES.keys())):
            toddler_warnings.append("Contains Southampton 6 azo dyes: proven correlation with clinical hyperactivity, tantrums, and ADHD in young children.")
        if any(e in detected_e_numbers for e in ["E950", "E951", "E952", "E954", "E955"]):
            toddler_warnings.append("Intense non-nutritive sweeteners interfere with pediatric appetite regulation and microbiome development.")
            
        toddler_status = "SUITABLE" if len(toddler_warnings) == 0 else ("MODERATE CAUTION" if len(toddler_warnings) <= 2 else "UNSUITABLE / HIGH RISK")
        toddler_impact = "Balanced for young child growth and cognitive development." if not toddler_warnings else "Contains additives or refined sugars correlated with pediatric behavioral disturbances and dental caries."

        # 4. Very Old People / Geriatric Cohort
        elderly_warnings = []
        if sodium_mg > 450.0:
            elderly_warnings.append(f"High sodium ({sodium_mg:.0f}mg/100g) accelerates arterial stiffening, left ventricular hypertrophy, and fluid retention.")
        if saturated_fat_g > 5.0:
            elderly_warnings.append(f"Elevated saturated fat ({saturated_fat_g:.1f}g/100g) aggravates cerebral and coronary atherosclerotic plaque progression.")
        if sugars_g > 15.0:
            elderly_warnings.append(f"High glycemic load ({sugars_g:.1f}g/100g) worsens microvascular circulation, frailty, and cognitive decline.")
        if any(e in detected_e_numbers for e in self.PHOSPHATE_ADDITIVES):
            elderly_warnings.append("Inorganic phosphorus additives accelerate vascular calcification in age-declined renal clearance.")
            
        elderly_status = "SAFE" if len(elderly_warnings) == 0 else ("MODERATE CAUTION" if len(elderly_warnings) <= 2 else "RESTRICTED / HIGH RISK")
        elderly_impact = "Mild metabolic impact compatible with senior adult vascular resilience." if not elderly_warnings else "Strains age-declined glomerular filtration, accelerates arterial stiffening, and spikes glycemic indices."

        return {
            "pregnant_mothers": {"status": preg_status, "impact": preg_impact, "warnings": preg_warnings},
            "babies_under_1yr": {"status": baby_status, "impact": baby_impact, "warnings": baby_warnings},
            "infants_toddlers": {"status": toddler_status, "impact": toddler_impact, "warnings": toddler_warnings},
            "very_old_people": {"status": elderly_status, "impact": elderly_impact, "warnings": elderly_warnings}
        }

    def audit_ncd_restrictions(
        self,
        ingredient_text: str,
        sugars_g: float,
        sodium_mg: float,
        saturated_fat_g: float,
        detected_e_numbers: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates clinical risks for key Non-Communicable Diseases (NCDs):
        Diabetes Mellitus, Hypertension, CVD, and Chronic Kidney Disease (CKD).
        """
        text_lower = ingredient_text.lower()
        
        # 1. Diabetes Mellitus
        dm_warnings = []
        if sugars_g > 15.0:
            dm_warnings.append(f"High sugar concentration ({sugars_g:.1f}g/100g) causes rapid postprandial glucose spiking.")
        elif sugars_g > 5.0:
            dm_warnings.append(f"Moderate sugar ({sugars_g:.1f}g/100g) requires carbohydrate counting and bolus adjustment.")
        if any(term in text_lower for term in ["maltodextrin", "glucose syrup", "corn syrup", "dextrose"]):
            dm_warnings.append("Contains ultra-high glycemic index refined starches/syrups (GI > 85).")
            
        dm_risk = "HIGH" if sugars_g > 15.0 or len(dm_warnings) >= 2 else ("MODERATE" if len(dm_warnings) > 0 else "LOW")
        
        # 2. Hypertension
        htn_warnings = []
        if sodium_mg > 600.0:
            htn_warnings.append(f"High sodium ({sodium_mg:.0f}mg/100g) directly elevates vascular peripheral resistance.")
        elif sodium_mg > 300.0:
            htn_warnings.append(f"Moderate sodium ({sodium_mg:.0f}mg/100g) contributes to daily salt ceiling (WHO < 2000mg/day).")
        if any(e in detected_e_numbers for e in ["E621", "E627", "E631"]):
            htn_warnings.append("Contains hidden sodium donors from glutamates and ribonucleotide flavor enhancers.")
            
        htn_risk = "HIGH" if sodium_mg > 600.0 else ("MODERATE" if sodium_mg > 300.0 or len(htn_warnings) > 0 else "LOW")
        
        # 3. Cardiovascular Disease (CVD)
        cvd_warnings = []
        if saturated_fat_g > 5.0:
            cvd_warnings.append(f"Saturated fat ({saturated_fat_g:.1f}g/100g) exceeds recommended atherogenic thresholds.")
        if any(term in text_lower for term in ["palm oil", "palm olein", "partially hydrogenated", "shortening"]):
            cvd_warnings.append("Contains atherogenic saturated/trans palm oil fractions or hydrogenated fats.")
            
        cvd_risk = "HIGH" if saturated_fat_g > 8.0 or "partially hydrogenated" in text_lower else ("MODERATE" if len(cvd_warnings) > 0 else "LOW")
        
        # 4. Chronic Kidney Disease (CKD)
        ckd_warnings = []
        if sodium_mg > 400.0:
            ckd_warnings.append(f"Excess sodium ({sodium_mg:.0f}mg/100g) precipitates fluid retention and glomerular hyperfiltration.")
        has_phosphates = any(e in detected_e_numbers for e in self.PHOSPHATE_ADDITIVES) or ("phosphate" in text_lower)
        if has_phosphates:
            ckd_warnings.append("Contains bioavailable inorganic phosphorus additives (absorption rate > 90%), worsening renal osteodystrophy.")
        if any(e in detected_e_numbers for e in ["E508", "E224"]) or ("potassium" in text_lower):
            ckd_warnings.append("Contains added potassium salts posing life-threatening hyperkalemia risk in stages 3-5 CKD.")
            
        ckd_risk = "HIGH" if has_phosphates or sodium_mg > 600.0 else ("MODERATE" if len(ckd_warnings) > 0 else "LOW")
        
        return {
            "diabetes": {"risk": dm_risk, "warnings": dm_warnings},
            "hypertension": {"risk": htn_risk, "warnings": htn_warnings},
            "cardiovascular": {"risk": cvd_risk, "warnings": cvd_warnings},
            "chronic_kidney": {"risk": ckd_risk, "warnings": ckd_warnings}
        }

    def audit_frequency_impact(
        self,
        predicted_grade: str,
        sugars_g: float,
        sodium_mg: float,
        saturated_fat_g: float,
        additives_count: int,
        is_ultra_processed: bool
    ) -> Dict[str, Any]:
        """
        Projects systemic physiological outcomes associated with frequent/daily vs.
        occasional consumption.
        """
        if predicted_grade in ["A", "B"]:
            frequency_recommendation = "Daily / Regular Consumption Feasible"
            chronic_risks = [
                "Nutrient-dense baseline compatible with balanced dietary guidelines.",
                "Minimal metabolic strain on insulin sensitivity and lipid clearance.",
                "Low additive burden promotes stable gut microbiome homeostasis."
            ]
            risk_tier = "MINIMAL"
        elif predicted_grade in ["C", "D"]:
            frequency_recommendation = "Moderate / 2-3 Times Weekly Maximum"
            chronic_risks = [
                "Cumulative sugar or sodium exposure may insidiously raise baseline blood pressure over months.",
                "Sub-optimal fiber-to-energy ratio may promote minor adipose tissue accumulation.",
                "Regular intake of emulsifiers or preservatives may modestly alter intestinal mucosal integrity."
            ]
            risk_tier = "MODERATE"
        else: # Grade E
            frequency_recommendation = "Strictly Occasional / Rare Indulgence (< Once Weekly)"
            chronic_risks = [
                "Chronic daily ingestion strongly correlated with hepatic steatosis (NAFLD) and insulin resistance.",
                "Sustained sodium load triggers vascular endothelial dysfunction and systemic hypertension.",
                "High ultra-processed additive cocktail significantly alters intestinal microbiota and elevates systemic CRP (inflammatory marker)."
            ]
            risk_tier = "SEVERE"
            
        return {
            "frequency_recommendation": frequency_recommendation,
            "risk_tier": risk_tier,
            "chronic_risks": chronic_risks
        }

    def compute_composite_health_rating(
        self,
        predicted_grade: str,
        moh_lights: Dict[str, Dict[str, str]],
        alarming_audit: Dict[str, Any],
        ncd_audit: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes ML predicted grade, SL MOH traffic lights, and chemical safety checks
        into an actionable 3-tier color scale: GREEN (Safe), ORANGE (Moderate Caution), RED (Critical Risk).
        """
        red_lights_count = sum(1 for item in moh_lights.values() if item["color"] == "Red")
        has_critical_toxicant = (
            len(alarming_audit["regulatory_discrepancies"]) > 0 or
            alarming_audit["benzene_risk"] or
            any(item["severity"] == "HIGH" for item in alarming_audit["carcinogens_and_toxins"])
        )
        high_ncd_risks = sum(1 for item in ncd_audit.values() if item["risk"] == "HIGH")
        
        # Decision Logic for Overall Color Rating
        if has_critical_toxicant or predicted_grade == "E" or red_lights_count >= 2 or high_ncd_risks >= 3:
            rating = "RED"
            status_title = "CRITICAL HEALTH & SAFETY RISK"
            summary = "This product exhibits substantial nutritional penalties, critical additive hazards, or prohibited regulatory discrepancies. Daily or regular consumption is strongly discouraged."
        elif predicted_grade in ["C", "D"] or red_lights_count == 1 or high_ncd_risks >= 1 or len(alarming_audit["southampton_dyes"]) > 0:
            rating = "ORANGE"
            status_title = "MODERATE CAUTION REQUIRED"
            summary = "Product possesses intermediate nutritional quality or notable additive markers. Consume with portion control and moderation, especially for vulnerable clinical groups."
        else:
            rating = "GREEN"
            status_title = "SAFE & HEALTHY PROFILE"
            summary = "Product demonstrates a wholesome nutritional profile with minimal negative nutrients and clean ingredient composition conforming to safety guidelines."
            
        return {
            "color_rating": rating,
            "status_title": status_title,
            "summary": summary
        }

    def full_audit(
        self,
        product_name: str,
        energy_kcal: float,
        sugars_g: float,
        fat_g: float,
        saturated_fat_g: float,
        sodium_mg: float,
        proteins_g: float,
        fiber_g: float,
        fruits_veg_pct: float,
        ingredients_text: str,
        predicted_grade: str,
        is_beverage: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-dimensional safety audit.
        """
        # 1. MOH Traffic Lights
        moh_lights = self.evaluate_moh_traffic_lights(sugars_g, sodium_mg, fat_g, is_beverage)
        
        # 2. Additive & Toxicological audit
        alarming_audit = self.audit_alarming_ingredients(ingredients_text)
        detected_e_nums = alarming_audit["detected_e_numbers"]
        
        # 3. Demographic suitability
        demo_audit = self.audit_demographic_suitability(
            ingredients_text, sugars_g, sodium_mg, saturated_fat_g, detected_e_nums
        )
        
        # 4. Clinical NCD risks
        ncd_audit = self.audit_ncd_restrictions(
            ingredients_text, sugars_g, sodium_mg, saturated_fat_g, detected_e_nums
        )
        
        # 5. Frequency impact
        freq_audit = self.audit_frequency_impact(
            predicted_grade, sugars_g, sodium_mg, saturated_fat_g, len(detected_e_nums), is_ultra_processed=len(detected_e_nums) > 2
        )
        
        # 6. Composite rating
        composite = self.compute_composite_health_rating(
            predicted_grade, moh_lights, alarming_audit, ncd_audit
        )
        
        return {
            "product_name": product_name,
            "predicted_grade": predicted_grade,
            "moh_traffic_lights": moh_lights,
            "alarming_audit": alarming_audit,
            "demographic_audit": demo_audit,
            "ncd_audit": ncd_audit,
            "frequency_audit": freq_audit,
            "composite_rating": composite
        }


if __name__ == "__main__":
    engine = SafetyAuditEngine()
    test_audit = engine.full_audit(
        product_name="Sample Sri Lankan Spicy Noodles",
        energy_kcal=465.0,
        sugars_g=3.5,
        fat_g=20.0,
        saturated_fat_g=9.8,
        sodium_mg=1150.0,
        proteins_g=9.0,
        fiber_g=2.2,
        fruits_veg_pct=2.0,
        ingredients_text="Wheat flour, palm olein, salt, chili paste, monosodium glutamate (INS 621), disodium ribonucleotides (INS 635), caramel colour (INS 150c), preservative (INS 211), ascorbic acid (INS 300).",
        predicted_grade="D",
        is_beverage=False
    )
    print("Full Audit Successfully Executed.")
    print("Composite Rating:", test_audit["composite_rating"])
    print("Benzene Risk:", test_audit["alarming_audit"]["benzene_risk"])
    print("NCD Risks:", {k: v["risk"] for k, v in test_audit["ncd_audit"].items()})
