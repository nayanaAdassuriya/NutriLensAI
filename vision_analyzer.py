"""
Vision and Granular Ingredient Analysis Module for NutriLens AI.

This module provides:
1. Multi-photo OCR text extraction using Windows Native OCR or PIL/Tesseract.
2. Inbuilt product matching: Identifies known local Sri Lankan supermarket items
   from packaging photos using fuzzy token matching and brand recognition.
3. Granular ingredient-by-ingredient audit: Deconstructs ingredient lists into
   individual components and audits each for functional class, health safety tier,
   clinical contraindications, and international regulatory discrepancy.
"""

import os
import re
import tempfile
import subprocess
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from data_pipeline import extract_e_numbers


# Comprehensive Knowledge Base of Specific Food Ingredients and Additives
INGREDIENT_KNOWLEDGE_BASE = {
    # Functional Fats & Oils
    "palm oil": {
        "category": "Refined Lipid",
        "risk_tier": "CAUTION",
        "function": "Cooking fat and structural frying oil",
        "clinical_impact": "High in palmitic acid (~45% saturated fat); promotes atherogenic LDL cholesterol accumulation when consumed regularly.",
        "demographic_warning": "Limit intake in individuals with cardiovascular disease or dyslipidemia.",
        "regulatory_status": "Permitted globally; subject to high saturated fat warnings in EU/UK front-of-pack labels."
    },
    "palm olein": {
        "category": "Refined Lipid",
        "risk_tier": "CAUTION",
        "function": "Fractionated liquid palm oil for frying",
        "clinical_impact": "High saturated fatty acid content; contributes to atherosclerotic plaque progression in coronary vessels.",
        "demographic_warning": "Caution for elderly adults and cardiac patients.",
        "regulatory_status": "Permitted worldwide."
    },
    "hydrogenated": {
        "category": "Industrial Trans-Fat",
        "risk_tier": "HIGH HAZARD",
        "function": "Solidified vegetable shortening",
        "clinical_impact": "Direct source of trans-fatty acids; strongly associated with endothelial inflammation, coronary heart disease, and insulin resistance.",
        "demographic_warning": "Contraindicated for all populations, especially CVD and hypertensive patients.",
        "regulatory_status": "Severely restricted or banned in EU, US FDA (partially hydrogenated oils no longer GRAS)."
    },
    "vegetable shortening": {
        "category": "Processed Lipid",
        "risk_tier": "CAUTION",
        "function": "Bakery texture improver and crumb softener",
        "clinical_impact": "High saturated and potential trans-fat fractions.",
        "demographic_warning": "Not recommended for regular consumption.",
        "regulatory_status": "Permitted."
    },

    # Sugars & Sweeteners
    "sugar": {
        "category": "Free Sugar / Simple Carbohydrate",
        "risk_tier": "CAUTION",
        "function": "Caloric sweetener and preservative",
        "clinical_impact": "Rapidly absorbed sucrose; triggers postprandial glycemic spikes, hyperinsulinemia, dental caries, and hepatic steatosis.",
        "demographic_warning": "Restricted in Diabetes Mellitus; WHO strongly advises zero added sugar for infants < 2 years.",
        "regulatory_status": "Permitted, subject to traffic-light sugar warning labels in Sri Lanka."
    },
    "cane sugar": {
        "category": "Free Sugar",
        "risk_tier": "CAUTION",
        "function": "Sweetener",
        "clinical_impact": "Elevates glycemic index; promotes insulin resistance upon chronic overconsumption.",
        "demographic_warning": "Contraindicated for strict glycemic control in Type 1 & 2 Diabetes.",
        "regulatory_status": "Permitted."
    },
    "liquid glucose": {
        "category": "Refined Syrup",
        "risk_tier": "CAUTION",
        "function": "Texture agent, humectant, and sweetener",
        "clinical_impact": "Extremely high Glycemic Index (GI > 90); acute blood glucose excursion hazard.",
        "demographic_warning": "High risk for diabetic patients.",
        "regulatory_status": "Permitted."
    },
    "dextrose": {
        "category": "Simple Sugar",
        "risk_tier": "CAUTION",
        "function": "Rapidly fermentable sweetener",
        "clinical_impact": "Pure D-glucose; induces immediate insulin secretion.",
        "demographic_warning": "Hazard for diabetic individuals.",
        "regulatory_status": "Permitted."
    },
    "aspartame": {
        "category": "Intense Artificial Sweetener",
        "code": "E951",
        "risk_tier": "CAUTION",
        "function": "Low-calorie intense sweetening agent",
        "clinical_impact": "Classified as IARC Group 2B (Possibly carcinogenic to humans). Hydrolyzes into phenylalanine, aspartic acid, and methanol.",
        "demographic_warning": "Strictly contraindicated for individuals with Phenylketonuria (PKU); crosses the placental barrier.",
        "regulatory_status": "Permitted with mandatory PKU warning declarations."
    },
    "acesulfame potassium": {
        "category": "Intense Sweetener",
        "code": "E950",
        "risk_tier": "CAUTION",
        "function": "Heat-stable synthetic sweetener",
        "clinical_impact": "Potential disruption of intestinal microbiome diversity; non-metabolizable xenobiotic.",
        "demographic_warning": "Avoid in infants and young children.",
        "regulatory_status": "Permitted with acceptable daily intake (ADI) limits."
    },

    # Preservatives & Chemical Agents
    "sodium benzoate": {
        "category": "Chemical Preservative",
        "code": "E211",
        "risk_tier": "CAUTION",
        "function": "Antimicrobial preservative targeting yeasts and molds",
        "clinical_impact": "Synergistically reacts with Ascorbic Acid (Vitamin C / E300) in acidic solutions to form toxic Benzene (Group 1 Carcinogen).",
        "demographic_warning": "May aggravate asthma, urticaria, and hyperactivity in pediatric patients.",
        "regulatory_status": "Strict concentration caps across EU (maximum 150 mg/kg) and Codex standards."
    },
    "potassium sorbate": {
        "category": "Preservative",
        "code": "E202",
        "risk_tier": "SAFE",
        "function": "Antifungal and antimicrobial agent",
        "clinical_impact": "Metabolized naturally by fatty acid beta-oxidation into water and carbon dioxide.",
        "demographic_warning": "Generally safe; rare contact allergic sensitivities reported.",
        "regulatory_status": "Widely authorized across global jurisdictions."
    },
    "potassium bromate": {
        "category": "Flour Treatment Agent",
        "code": "E924",
        "risk_tier": "HIGH HAZARD",
        "function": "Dough conditioner and flour bleaching agent",
        "clinical_impact": "Potent genotoxic oxidizer and classified as IARC Group 2B human renal carcinogen.",
        "demographic_warning": "Severe health hazard for all age groups.",
        "regulatory_status": "Banned in the European Union, United Kingdom, Canada, and Sri Lanka."
    },
    "titanium dioxide": {
        "category": "Food Colour / Whitener",
        "code": "E171",
        "risk_tier": "HIGH HAZARD",
        "function": "White pigment and opacity enhancer",
        "clinical_impact": "Nanoparticle accumulation in lymphoid tissue; exhibits genotoxicity and chromosomal break hazards in gut mucosal lining.",
        "demographic_warning": "Unsafe for human consumption according to EFSA guidance.",
        "regulatory_status": "Completely banned in the European Union (EU Reg 2022/63); lagging in certain regional standards."
    },

    # Flavor Enhancers & Salts
    "salt": {
        "category": "Mineral / Sodium Donor",
        "risk_tier": "CAUTION",
        "function": "Flavoring agent and preservative",
        "clinical_impact": "High sodium intake expands extracellular fluid volume, raises systemic arterial blood pressure, and strains kidney nephrons.",
        "demographic_warning": "Strict limitation in Hypertension, Chronic Kidney Disease, and Infant nutrition.",
        "regulatory_status": "Subject to Sri Lankan MOH Red traffic light if sodium exceeds 500 mg / 100g."
    },
    "iodized salt": {
        "category": "Fortified Mineral",
        "risk_tier": "CAUTION",
        "function": "Essential micronutrient fortification (iodine) and seasoning",
        "clinical_impact": "Beneficial for thyroid health in moderation; excess sodium load retains cardiovascular hazard.",
        "demographic_warning": "Monitor in hypertensive patients.",
        "regulatory_status": "Mandated iodization under Sri Lankan Food Regulations."
    },
    "monosodium glutamate": {
        "category": "Flavor Enhancer",
        "code": "E621",
        "risk_tier": "CAUTION",
        "function": "Umami savory flavor enhancer",
        "clinical_impact": "Excitotoxic neurotransmitter trigger; adds hidden sodium; may provoke transient bronchospasm and flushing in sensitive individuals.",
        "demographic_warning": "Contraindicated in individuals with glutamate intolerance or severe uncontrolled hypertension.",
        "regulatory_status": "Permitted with mandatory ingredient disclosure."
    },
    "disodium 5'-ribonucleotides": {
        "category": "Flavor Enhancer",
        "code": "E635",
        "risk_tier": "CAUTION",
        "function": "Potentiates savory taste in synergy with glutamates",
        "clinical_impact": "Metabolized to purines; can trigger acute hyperuricemia or gout exacerbation.",
        "demographic_warning": "Avoid in patients with gout, hyperuricemia, or severe renal impairment.",
        "regulatory_status": "Authorized globally."
    },

    # Synthetic Azo Colors (Southampton 6)
    "tartrazine": {
        "category": "Synthetic Azo Dye",
        "code": "E102",
        "risk_tier": "CAUTION",
        "function": "Artificial lemon-yellow colorant",
        "clinical_impact": "Linked to neurobehavioral excitability, ADHD symptoms, histaminic reactions, and pediatric attention disruption.",
        "demographic_warning": "Mandatory warning required for children in the EU; contraindicated in aspirin-sensitive asthma.",
        "regulatory_status": "Permitted in Sri Lanka; EU mandates warning: 'May have an adverse effect on activity and attention in children'."
    },
    "sunset yellow": {
        "category": "Synthetic Azo Dye",
        "code": "E110",
        "risk_tier": "CAUTION",
        "function": "Artificial orange food color",
        "clinical_impact": "Southampton study dye associated with hyperactivity in children and occasional urticaria.",
        "demographic_warning": "Unsuitable for infants and hyperactive children.",
        "regulatory_status": "Subject to mandatory warning labels in EU."
    },
    "carmoisine": {
        "category": "Synthetic Azo Dye",
        "code": "E122",
        "risk_tier": "CAUTION",
        "function": "Artificial red coloring agent",
        "clinical_impact": "Azo colorant provoking pseudo-allergic responses, skin rashes, and child attention impairment.",
        "demographic_warning": "Restricted in pediatric foods.",
        "regulatory_status": "Banned in USA/Canada; strictly warned in the EU."
    },
    "caramel colour": {
        "category": "Colouring Agent",
        "code": "E150c",
        "risk_tier": "CAUTION",
        "function": "Brown coloring agent derived from heated carbohydrates",
        "clinical_impact": "Ammonia caramel contains 4-methylimidazole (4-MEI), an impurity classified by IARC as possibly carcinogenic (Group 2B).",
        "demographic_warning": "Moderate intake advised.",
        "regulatory_status": "Permitted with strict limits on 4-MEI residual levels."
    },

    # Common Wholesome / Benign Ingredients
    "wheat flour": {
        "category": "Cereal Carbohydrate / Staple",
        "risk_tier": "SAFE",
        "function": "Primary structural bakery matrix",
        "clinical_impact": "Source of complex starch and plant protein; contains gluten.",
        "demographic_warning": "Contraindicated in Celiac Disease and gluten sensitivity.",
        "regulatory_status": "Standard food staple."
    },
    "whole wheat flour": {
        "category": "Whole Grain",
        "risk_tier": "SAFE",
        "function": "Nutrient-dense cereal base",
        "clinical_impact": "High in insoluble dietary fiber; slows glucose absorption and enhances bowel regularity.",
        "demographic_warning": "Excellent for diabetes management and cardiovascular longevity.",
        "regulatory_status": "Highly recommended in global dietary guidelines."
    },
    "corn": {
        "category": "Whole Grain / Starch",
        "risk_tier": "SAFE",
        "function": "Energy and grain base",
        "clinical_impact": "Provides carotenoids (lutein, zeaxanthin) and dietary fiber.",
        "demographic_warning": "Wholesome for all age groups.",
        "regulatory_status": "Permitted staple."
    },
    "soya bean": {
        "category": "Legume / Plant Protein",
        "risk_tier": "SAFE",
        "function": "High-biological-value protein and isoflavones",
        "clinical_impact": "Cardioprotective lipid profile; low glycemic index.",
        "demographic_warning": "Major allergen; requires declared allergen labeling.",
        "regulatory_status": "Permitted."
    },
    "green gram": {
        "category": "Traditional Legume",
        "risk_tier": "SAFE",
        "function": "Plant protein, minerals, and soluble fiber",
        "clinical_impact": "Low GI, rich in potassium, iron, and prebiotic resistant starch; ideal for NCD mitigation.",
        "demographic_warning": "Safe and recommended across all life stages.",
        "regulatory_status": "Traditional food staple in Sri Lankan Food Composition Database."
    },
    "rice": {
        "category": "Cereal Grain",
        "risk_tier": "SAFE",
        "function": "Starch base and staple energy",
        "clinical_impact": "Easily digestible and naturally hypoallergenic; low allergenicity.",
        "demographic_warning": "Safe for infants and geriatric diets.",
        "regulatory_status": "Standard national staple."
    },
    "yeast": {
        "category": "Microbial Culture",
        "risk_tier": "SAFE",
        "function": "Fermentation agent and natural B-vitamin donor",
        "clinical_impact": "Provides natural folates and niacin; promotes gut fermentation.",
        "demographic_warning": "Generally recognized as safe.",
        "regulatory_status": "Standard leavening agent."
    },
    "citric acid": {
        "category": "Acidity Regulator / Antioxidant",
        "code": "E330",
        "risk_tier": "SAFE",
        "function": "Tartness enhancer and natural preservative",
        "clinical_impact": "Intermediate in the Krebs cycle; fully metabolized by cellular respiration.",
        "demographic_warning": "Excessive direct exposure may contribute to tooth enamel demineralization.",
        "regulatory_status": "Permitted without quantitative limit (Quantum Satis)."
    },
    "ascorbic acid": {
        "category": "Vitamin / Antioxidant",
        "code": "E300",
        "risk_tier": "SAFE",
        "function": "Vitamin C nutrient and dough treatment agent",
        "clinical_impact": "Essential antioxidant. Caution: If combined with Sodium Benzoate (E211), benzene hazard occurs.",
        "demographic_warning": "Wholesome alone; avoid in formulas with benzoate preservatives.",
        "regulatory_status": "Permitted globally."
    },
    "soy lecithin": {
        "category": "Emulsifier",
        "code": "E322",
        "risk_tier": "SAFE",
        "function": "Lipid-water homogenizer and stabilizing agent",
        "clinical_impact": "Phospholipid supplying choline; supports cell membrane stability.",
        "demographic_warning": "Allergen warning required for soy-allergic individuals.",
        "regulatory_status": "Permitted worldwide."
    },
    "pectin": {
        "category": "Soluble Fiber / Gelling Agent",
        "code": "E440",
        "risk_tier": "SAFE",
        "function": "Fruit-derived gelling agent",
        "clinical_impact": "Acts as a prebiotic fiber supporting short-chain fatty acid synthesis in the colon.",
        "demographic_warning": "Safe and beneficial for digestive health.",
        "regulatory_status": "Permitted Quantum Satis."
    }
}


def run_windows_native_ocr(image_path: str) -> str:
    """
    Executes high-accuracy native Windows OCR (Windows.Media.Ocr.OcrEngine)
    on an image file via PowerShell Windows Runtime API.
    """
    ps_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_engine.ps1")
    if not os.path.exists(ps_script_path):
        return ""
        
    try:
        cmd = [
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", ps_script_path,
            "-ImagePath", image_path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception as e:
        print(f"[OCR] Native OCR execution notice: {e}")
        
    return ""


def extract_text_from_images(image_files: List[Any]) -> str:
    """
    Extracts and compiles text from one or more uploaded image files (PIL Images or Streamlit UploadedFiles).
    """
    combined_text = []
    
    for idx, uploaded_file in enumerate(image_files):
        try:
            # Save uploaded image to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                if hasattr(uploaded_file, "read"):
                    tmp.write(uploaded_file.getvalue())
                else:
                    # PIL image object
                    uploaded_file.save(tmp.name)
                temp_path = tmp.name
                
            # Perform OCR extraction
            extracted = run_windows_native_ocr(temp_path)
            if extracted:
                combined_text.append(f"--- Photo #{idx+1} Recognized Text ---\n{extracted}")
                
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as err:
            print(f"[OCR] Error processing image #{idx+1}: {err}")
            
    return "\n\n".join(combined_text)


def match_inbuilt_product(ocr_text: str, catalog_df: pd.DataFrame) -> Tuple[Optional[pd.Series], float, List[str]]:
    """
    Identifies if packaging photos match a known inbuilt Sri Lankan supermarket item
    (Munchee, Maliban, Elephant House, Highland, Kotmale, Anchor, Prima, Samaposha, etc.).
    Returns:
        - Matched product row (Series or None)
        - Confidence score (0 to 100%)
        - Matched token rationale list
    """
    if not ocr_text or not ocr_text.strip() or catalog_df.empty:
        return None, 0.0, []
        
    text_lower = ocr_text.lower()
    
    best_match = None
    best_score = 0.0
    best_tokens = []
    
    # Check each unique product in catalog
    unique_products = catalog_df.drop_duplicates(subset=["brand", "product_name"]).copy()
    
    for _, row in unique_products.iterrows():
        score = 0.0
        tokens_found = []
        
        brand = str(row["brand"]).lower()
        prod_name = str(row["product_name"]).lower().split("(")[0].strip()
        category = str(row["category"]).lower()
        
        # 1. Brand match (Strong weight)
        if brand in text_lower:
            score += 45.0
            tokens_found.append(f"Brand detected: '{row['brand']}'")
            
        # 2. Key product name tokens (e.g. 'cream cracker', 'lemon puff', 'ginger beer')
        name_words = [w for w in re.split(r'[\s\-]+', prod_name) if len(w) > 3]
        matched_words = [w for w in name_words if w in text_lower]
        
        if matched_words:
            word_fraction = len(matched_words) / max(len(name_words), 1)
            score += word_fraction * 45.0
            tokens_found.append(f"Keywords matched: {', '.join(matched_words)}")
            
        # 3. Characteristic ingredients
        ing_snippet = str(row.get("ingredients_text", "")).lower()
        if "amla" in text_lower and "amla" in ing_snippet:
            score += 15.0
            tokens_found.append("Specific ingredient matched: 'amla / nelli'")
        if "kurakkan" in text_lower and "kurakkan" in ing_snippet:
            score += 15.0
            tokens_found.append("Specific ingredient matched: 'kurakkan'")
        if "kottu" in text_lower and "kottu" in prod_name:
            score += 25.0
            tokens_found.append("Specialty product matched: 'Kottu Mee'")
            
        score = min(score, 99.0)
        
        if score > best_score:
            best_score = score
            best_match = row
            best_tokens = tokens_found
            
    # Threshold for considering it an identified inbuilt product
    if best_score >= 40.0:
        return best_match, round(best_score, 1), best_tokens
    else:
        return None, round(best_score, 1), []


def clean_ingredients_statement(text: str) -> str:
    """
    Strips leading headers and truncates trailing non-ingredient boilerplate
    (e.g. allergen tags, social media handles, marketing slogans, URLs, contact numbers).
    """
    if not text or not isinstance(text, str):
        return ""
        
    cleaned = re.sub(r'^(?:ingredients|contains|composition|ingrediants|අමුද්‍රව්‍ය|தேவையான பொருட்கள்)\s*[:\-–]?\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Comprehensive stop pattern for non-ingredient trailing sections
    stop_pattern = r'\b(?:contains\s*[:\-]?\s*[a-z]+|contains\s+[a-z]+|may contain|allergen|allergy advice|allergy information|let[\'’]?s\s+(?:[a-z]\s+)?get social|get social|visit\s+[a-z0-9\.]+|email\s+[a-z0-9@\.]+|questions\s+or\s+comments|customer care|consumer relations|call\s+1-|1-800|distributed by|manufactured by|marketed by|produced by|packaged by|product of|packed for|best before|mfd|exp|lot\s*[:#]|batch\s*[:#]|net wt|net weight|nutrition facts|nutrition information|serving size|servings per|keep in a cool|store in a cool|refrigerate after|guaranteed fresh|quality guaranteed|satisfaction guaranteed)\b'
    m = re.search(stop_pattern, cleaned, re.IGNORECASE)
    if m:
        cleaned = cleaned[:m.start()].strip()
        
    return cleaned.rstrip('.,; ').strip()


def extract_ingredients_from_ocr(ocr_text: str) -> str:
    """
    Extracts the ingredient list statement from raw OCR text across multiple packaging photos.
    Uses flexible regex patterns to isolate 'INGREDIENTS' blocks, strips marketing/allergen
    chaff, and falls back to detecting food constituent sequences if no header keyword was found.
    """
    if not ocr_text or not ocr_text.strip():
        return ""
        
    # Strip photo header banners
    clean_text = re.sub(r'---\s*Photo\s*#\d+\s*Recognized Text\s*---', '', ocr_text)
    
    # 1. Primary Regex: Look for standard ingredients section header
    pattern = r'(?:ingred[a-z]{3,7}|composition|ingrediants|ingredients|අමුද්‍රව්‍ය|தேவையான பொருட்கள்)\s*[:\-–]?\s*([\s\S]+?)(?=(?:allergen|allergy advice|nutrition facts|nutrition declaration|nutritional|storage|store in|keep in|mfd|exp|best before|net weight|net wt|batch|reg no|mfg|manufactured|marketed|distributed|packed by|customer care|bar code|barcode|fssai|$))'
    match = re.search(pattern, clean_text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        extracted = re.sub(r'\s*\n\s*', ' ', extracted)
        extracted = re.sub(r'\s{2,}', ' ', extracted).strip()
        extracted = clean_ingredients_statement(extracted)
        if len(extracted) > 10:
            return extracted
            
    # 2. Secondary heuristic: scan line-by-line for ingredient-rich lines
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    candidate_lines = []
    found_start = False
    
    ingredient_keywords = [
        "flour", "sugar", "oil", "fat", "salt", "water", "milk", "syrup", "extract",
        "yeast", "cocoa", "malt", "acid", "starch", "emulsifier", "leavening",
        "preservative", "flavour", "flavor", "colour", "color", "ins", "e1", "e2",
        "e3", "e4", "e5", "e6", "e9", "wheat", "rice", "soya", "soy", "cream",
        "butter", "dextrose", "pectin", "lecithin", "benzoate", "tartrazine",
        "palm", "curry", "chilli", "garlic", "onion", "spice", "cereal", "almonds",
        "peanuts", "cashews", "hazelnuts", "walnuts", "seasoning", "paprika"
    ]
    
    stop_keywords = [
        "nutrition", "energy", "calories", "protein 1", "carbohydrate",
        "mfd", "exp", "net weight", "batch", "barcode", "manufactured by",
        "customer care", "get social", "visit", "email", "distributed by", "contains"
    ]
    
    for line in lines:
        line_lower = line.lower()
        if any(stop_kw in line_lower for stop_kw in stop_keywords):
            if found_start:
                break
            continue
            
        has_ing_kw = any(kw in line_lower for kw in ingredient_keywords)
        has_commas = ',' in line
        
        if has_ing_kw or (found_start and has_commas):
            found_start = True
            candidate_lines.append(line)
        elif found_start:
            if len(line) < 15 and not has_commas:
                break
            candidate_lines.append(line)
            
    if candidate_lines:
        combined = " ".join(candidate_lines)
        combined = re.sub(r'\s{2,}', ' ', combined).strip()
        return clean_ingredients_statement(combined)
        
    # 3. Fallback: if short, return the cleaned raw text
    return clean_ingredients_statement(clean_text)


def extract_nutrition_from_ocr(ocr_text: str) -> Dict[str, float]:
    """
    Extracts nutritional metric values from OCR text (Nutrition Facts panel).
    Applies strict biological boundaries (macronutrients <= 100g/100g) and differentiates
    caloric values (e.g. 'Calories from Fat: 120') from actual lipid weight in grams.
    """
    if not ocr_text:
        return {}
        
    results = {}
    
    def parse_numeric(pattern: str, text: str) -> Optional[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                return None
        return None
        
    energy = parse_numeric(r'(?:energy|calories|caloric value)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:kcal|kj)?', ocr_text)
    if energy is not None and energy > 0:
        results["energy_kcal"] = min(energy, 2000.0)

    # Total Fat: Avoid matching "Calories from fat: 120" or "Fat Calories"
    fat_val = parse_numeric(r'(?:total fat|lipids?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?', ocr_text)
    if fat_val is None:
        # Fallback to isolated "fat: X g" but skip if preceded by "calories from"
        m_fat = re.search(r'(?<!calories from\s)(?<!cal from\s)\bfat\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?', ocr_text, re.IGNORECASE)
        if m_fat:
            try:
                fat_val = float(m_fat.group(1))
            except (ValueError, TypeError):
                fat_val = None

    if fat_val is not None:
        # Sane check: fat cannot exceed 100g per 100g
        results["fat_g"] = min(max(fat_val, 0.0), 100.0) if fat_val <= 100.0 else 15.0

    sat_fat = parse_numeric(r'(?:saturated fat|saturates|sat fat)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?', ocr_text)
    if sat_fat is not None:
        results["saturated_fat_g"] = min(max(sat_fat, 0.0), 100.0) if sat_fat <= 100.0 else 5.0

    sugars = parse_numeric(r'(?:total sugars?|sugars?|of which sugars?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?', ocr_text)
    if sugars is not None:
        results["sugars_g"] = min(max(sugars, 0.0), 100.0) if sugars <= 100.0 else 10.0

    sodium = parse_numeric(r'(?:sodium)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*mg?', ocr_text)
    if sodium is None:
        salt_g = parse_numeric(r'(?:salt)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g', ocr_text)
        if salt_g is not None:
            sodium = round(salt_g * 400.0, 1)
    if sodium is not None:
        results["sodium_mg"] = min(max(sodium, 0.0), 10000.0)

    protein = parse_numeric(r'(?:proteins?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?', ocr_text)
    if protein is not None:
        results["proteins_g"] = min(max(protein, 0.0), 100.0) if protein <= 100.0 else 8.0

    fiber = parse_numeric(r'(?:dietary fiber|fibre|fiber)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?', ocr_text)
    if fiber is not None:
        results["fiber_g"] = min(max(fiber, 0.0), 100.0) if fiber <= 100.0 else 3.0
    
    return {k: v for k, v in results.items() if v is not None}


def parse_raw_ingredients_list(ingredients_text: str) -> List[str]:
    """
    Splits unstructured ingredient text into clean, individual component tokens.
    Handles nested parentheses, brackets, and filters out non-ingredient artifacts.
    """
    if not ingredients_text or not isinstance(ingredients_text, str):
        return []
        
    cleaned = clean_ingredients_statement(ingredients_text)
    
    # Split by commas or semicolons that are outside parentheses/brackets
    tokens = []
    current = []
    depth = 0
    
    for char in cleaned:
        if char in '([{':
            depth += 1
            current.append(char)
        elif char in ')]}':
            depth = max(0, depth - 1)
            current.append(char)
        elif char in [',', ';'] and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(char)
            
    if current:
        token = "".join(current).strip()
        if token:
            tokens.append(token)
            
    # Post-clean and discard non-ingredient chaff tokens
    filtered_tokens = []
    junk_patterns = [
        "social", "visit", "email", "website", "www.", "customer care",
        "distributed by", "manufactured by", "call 1-", "questions or",
        "contains ", "may contain", "nutrition", "servings per", "net wt"
    ]
    
    for t in tokens:
        t_clean = t.rstrip('.,; ').strip()
        t_lower = t_clean.lower()
        if not t_clean or len(t_clean) < 2:
            continue
        if any(jp in t_lower for jp in junk_patterns):
            continue
        filtered_tokens.append(t_clean)
            
    return filtered_tokens


def analyze_ingredient_item(raw_token: str) -> Dict[str, Any]:
    """
    Performs clinical, functional, and regulatory audit on a single ingredient token.
    """
    token_lower = raw_token.lower().strip()
    
    # Extract any E-number present
    e_numbers = extract_e_numbers(raw_token)
    e_code = e_numbers[0] if e_numbers else None
    
    # Search knowledge base for exact match or substring match
    matched_kb = None
    
    for key, data in INGREDIENT_KNOWLEDGE_BASE.items():
        if key in token_lower or (e_code and data.get("code") == e_code):
            matched_kb = data
            break
            
    if matched_kb:
        return {
            "ingredient_name": raw_token,
            "category": matched_kb["category"],
            "code": e_code or matched_kb.get("code", "—"),
            "risk_tier": matched_kb["risk_tier"],
            "function": matched_kb["function"],
            "clinical_impact": matched_kb["clinical_impact"],
            "demographic_warning": matched_kb.get("demographic_warning", "No specific vulnerable contraindications."),
            "regulatory_status": matched_kb.get("regulatory_status", "Permitted under standard food regulations.")
        }
    else:
        # Default fallback analysis for unlisted natural/common ingredients
        is_chemical = e_code is not None or any(term in token_lower for term in ["acid", "artificial", "synthetic", "preservative", "colour", "color"])
        tier = "CAUTION" if is_chemical else "SAFE"
        
        return {
            "ingredient_name": raw_token,
            "category": "Chemical Additive" if is_chemical else "Natural / Food Ingredient",
            "code": e_code or "—",
            "risk_tier": tier,
            "function": "Functional additive / formulating component" if is_chemical else "Nutritional constituent",
            "clinical_impact": "Synthetic food additive; limit cumulative ultra-processed consumption." if is_chemical else "Standard dietary food ingredient.",
            "demographic_warning": "Caution for young children and sensitive individuals." if is_chemical else "Generally recognized as safe.",
            "regulatory_status": "Regulated under Food (General) Regulations."
        }


def analyze_ingredients_granular(ingredients_text: str) -> List[Dict[str, Any]]:
    """
    Audits every single ingredient one by one and returns a list of audit profiles.
    """
    raw_tokens = parse_raw_ingredients_list(ingredients_text)
    if not raw_tokens and ingredients_text.strip():
        # Fallback split
        raw_tokens = [t.strip() for t in ingredients_text.split(",") if t.strip()]
        
    audit_results = []
    for token in raw_tokens:
        audit_results.append(analyze_ingredient_item(token))
        
    return audit_results


if __name__ == "__main__":
    sample_text = "Wheat flour, palm oil, salt, yeast, monosodium glutamate (INS 621), tartrazine (INS 102), sodium benzoate (INS 211)."
    res = analyze_ingredients_granular(sample_text)
    print(f"Parsed {len(res)} ingredients:")
    for item in res:
        print(f"[{item['risk_tier']}] {item['ingredient_name']} -> {item['category']} | {item['clinical_impact'][:60]}...")
