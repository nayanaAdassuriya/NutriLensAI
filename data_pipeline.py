"""
Data Pipeline Module for NutriLens AI.

This module is responsible for:
1. Synthesizing / loading a curated dataset of regional packaged food products
   representative of Sri Lankan supermarkets (Cargills Food City, Keells Super)
   and regional Open Food Facts records.
2. Parsing raw ingredient strings to detect additive codes (E-numbers / INS codes)
   and compute additive density.
3. Aligning nutritional attributes with the Sri Lankan Ministry of Health Food
   (Colour Coding for Sugar, Salt, and Fat) Regulations and international Nutri-Score standards.
4. Preprocessing features, handling missing tokens, and stratifying train-validation-test partitions.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
DATA_FILE_PATH = os.path.join(DATA_DIR, "local_packaged_foods_lk.csv")


def extract_e_numbers(ingredient_text: str) -> List[str]:
    """
    Extracts European (E-number) and International Numbering System (INS) additive codes
    from raw text using regular expressions.
    """
    if not isinstance(ingredient_text, str) or not ingredient_text.strip():
        return []
    
    # Matches patterns like E102, INS 621, INS450i, E-150d, INS 330
    pattern = r'\b(?:INS|E)[-\s]?([0-9]{3,4}[a-z]?)\b'
    matches = re.findall(pattern, ingredient_text, flags=re.IGNORECASE)
    
    normalized_codes = []
    for code in matches:
        normalized = f"E{code.upper().replace(' ', '')}"
        normalized_codes.append(normalized)
    
    return list(dict.fromkeys(normalized_codes))


def compute_health_grade(
    energy_kcal: float,
    sugars_g: float,
    saturated_fat_g: float,
    sodium_mg: float,
    proteins_g: float,
    fiber_g: float,
    fruits_veg_pct: float,
    is_beverage: bool = False
) -> str:
    """
    Computes a 5-tier nutritional health grade (A, B, C, D, E) grounded in
    the Rayner/FSANZ scoring algorithm reconciled with Sri Lankan Food (Colour Coding
    for Sugar, Salt and Fat) Regulations.
    
    Negative points are accumulated for energy, sugar, saturated fatty acids, and sodium.
    Positive points are earned for dietary fiber, protein, and fruit/vegetable percentage.
    """
    # Negative component points
    p_energy = int(np.clip(energy_kcal / 80.0, 0, 10))
    p_sugar = int(np.clip(sugars_g / 4.5, 0, 10)) if not is_beverage else int(np.clip(sugars_g / 1.5, 0, 10))
    p_sat_fat = int(np.clip(saturated_fat_g / 1.0, 0, 10))
    p_sodium = int(np.clip(sodium_mg / 90.0, 0, 10))
    
    points_negative = p_energy + p_sugar + p_sat_fat + p_sodium
    
    # Positive component points
    p_fiber = int(np.clip(fiber_g / 0.7, 0, 5))
    p_protein = int(np.clip(proteins_g / 1.6, 0, 5))
    p_fv = int(np.clip(fruits_veg_pct / 20.0, 0, 5))
    
    points_positive = p_fiber + p_protein + p_fv
    
    final_score = points_negative - points_positive
    
    if final_score <= -1:
        return 'A'
    elif final_score <= 2:
        return 'B'
    elif final_score <= 10:
        return 'C'
    elif final_score <= 18:
        return 'D'
    else:
        return 'E'


def generate_local_food_dataset(filepath: str = DATA_FILE_PATH) -> pd.DataFrame:
    """
    Generates and persists a structured local benchmark dataset of Sri Lankan retail
    and regional packaged food products. Covers staple consumer goods across supermarkets
    like Cargills Food City and Keells Super.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    records = [
        # Dairy & Milks
        {"brand": "Highland", "product_name": "Pasteurised Fresh Full Cream Milk", "category": "Dairy", "energy_kcal": 65.0, "sugars_g": 4.8, "fat_g": 3.6, "saturated_fat_g": 2.2, "sodium_mg": 45.0, "proteins_g": 3.2, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 1, "ingredients_text": "100% pure fresh cow's milk."},
        {"brand": "Pelwatte", "product_name": "Non-Fat Skimmed Milk", "category": "Dairy", "energy_kcal": 36.0, "sugars_g": 4.9, "fat_g": 0.2, "saturated_fat_g": 0.1, "sodium_mg": 50.0, "proteins_g": 3.4, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 1, "ingredients_text": "Pasteurised skimmed milk, vitamin A, vitamin D3."},
        {"brand": "Anchor", "product_name": "Full Cream Milk Powder", "category": "Dairy", "energy_kcal": 510.0, "sugars_g": 38.0, "fat_g": 28.5, "saturated_fat_g": 17.5, "sodium_mg": 380.0, "proteins_g": 24.0, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Whole milk powder, emulsifier (INS 322 soy lecithin)."},
        {"brand": "Highland", "product_name": "Sweetened Set Curd & Treacle Cup", "category": "Dairy", "energy_kcal": 145.0, "sugars_g": 18.2, "fat_g": 4.2, "saturated_fat_g": 2.6, "sodium_mg": 65.0, "proteins_g": 4.1, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Pasteurised buffalo milk, starter culture, kithul treacle, sucrose, stabilizer (INS 440)."},
        {"brand": "Kotmale", "product_name": "Vanilla Flavoured UHT Milk", "category": "Dairy", "energy_kcal": 82.0, "sugars_g": 11.5, "fat_g": 2.2, "saturated_fat_g": 1.4, "sodium_mg": 58.0, "proteins_g": 3.0, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 1, "ingredients_text": "Cow's milk solids, cane sugar, permitted food colour (INS 102), artificial vanilla flavour, stabilizer (INS 407)."},
        {"brand": "Kotmale", "product_name": "Pasteurized Processed Cheddar Cheese", "category": "Dairy", "energy_kcal": 330.0, "sugars_g": 1.0, "fat_g": 26.0, "saturated_fat_g": 16.5, "sodium_mg": 780.0, "proteins_g": 21.0, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Cheddar cheese (milk, salt, cultures, rennet), emulsifying salts (INS 339, INS 452), preservative (INS 200)."},

        # Biscuits & Bakery (Munchee, Maliban, CBL)
        {"brand": "Munchee", "product_name": "Super Cream Cracker", "category": "Biscuits & Bakery", "energy_kcal": 450.0, "sugars_g": 1.8, "fat_g": 14.5, "saturated_fat_g": 6.8, "sodium_mg": 560.0, "proteins_g": 9.5, "fiber_g": 3.8, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Wheat flour, vegetable fat (palm oil), malt extract, salt, yeast, raising agents (INS 500ii, INS 503ii), emulsifier (INS 322)."},
        {"brand": "Maliban", "product_name": "Lemon Puff Sandwich Biscuit", "category": "Biscuits & Bakery", "energy_kcal": 490.0, "sugars_g": 31.5, "fat_g": 21.0, "saturated_fat_g": 11.2, "sodium_mg": 280.0, "proteins_g": 5.2, "fiber_g": 1.2, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Wheat flour, sugar, vegetable oil (palm), dextrose, citric acid (INS 330), synthetic food color (INS 102 tartrazine), natural lemon oil, raising agents (INS 500ii, INS 503ii)."},
        {"brand": "Munchee", "product_name": "Chocolate Marie Biscuits", "category": "Biscuits & Bakery", "energy_kcal": 465.0, "sugars_g": 24.0, "fat_g": 15.5, "saturated_fat_g": 7.4, "sodium_mg": 310.0, "proteins_g": 7.1, "fiber_g": 2.4, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Wheat flour, sugar, vegetable fat (palm), cocoa powder, milk solids, raising agent (INS 500ii), artificial vanillin."},
        {"brand": "Maliban", "product_name": "Ginger Biscuits", "category": "Biscuits & Bakery", "energy_kcal": 448.0, "sugars_g": 26.5, "fat_g": 14.0, "saturated_fat_g": 6.9, "sodium_mg": 390.0, "proteins_g": 6.0, "fiber_g": 2.1, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Wheat flour, sugar, vegetable shortening, natural ginger extract, salt, raising agents (INS 500ii, INS 503ii), colour (INS 150d caramel)."},
        {"brand": "Munchee", "product_name": "Tikiri Marie", "category": "Biscuits & Bakery", "energy_kcal": 435.0, "sugars_g": 21.0, "fat_g": 11.8, "saturated_fat_g": 5.5, "sodium_mg": 340.0, "proteins_g": 7.8, "fiber_g": 2.0, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Wheat flour, sugar, vegetable oil, liquid glucose, corn flour, salt, leavening agent (INS 500ii), emulsifier (INS 322)."},
        {"brand": "Maliban", "product_name": "Real Bran Whole Wheat Cracker", "category": "Biscuits & Bakery", "energy_kcal": 410.0, "sugars_g": 3.0, "fat_g": 11.0, "saturated_fat_g": 4.5, "sodium_mg": 420.0, "proteins_g": 11.0, "fiber_g": 8.5, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Whole wheat flour, wheat bran, vegetable fat, oat fiber, sesame seeds, salt, yeast, raising agent (INS 500ii)."},

        # Cereals & Grains
        {"brand": "CBL Plenty Foods", "product_name": "Samaposha Traditional Cereal Blend", "category": "Cereals & Breakfast", "energy_kcal": 380.0, "sugars_g": 4.5, "fat_g": 6.2, "saturated_fat_g": 1.5, "sodium_mg": 40.0, "proteins_g": 18.5, "fiber_g": 10.2, "fruits_veg_pct": 20.0, "is_beverage": 0, "ingredients_text": "Corn, soya bean, green gram, rice, vitamins (A, B1, B2, B3, B6, B12, C, D3, E), minerals (iron, calcium)."},
        {"brand": "Kellogg's", "product_name": "Corn Flakes Original", "category": "Cereals & Breakfast", "energy_kcal": 378.0, "sugars_g": 8.0, "fat_g": 0.9, "saturated_fat_g": 0.2, "sodium_mg": 480.0, "proteins_g": 7.0, "fiber_g": 3.0, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Milled corn, sugar, malt extract, iodized salt, vitamins and minerals."},
        {"brand": "Nestle", "product_name": "Koko Krunch Chocolate Cereal", "category": "Cereals & Breakfast", "energy_kcal": 390.0, "sugars_g": 28.0, "fat_g": 3.5, "saturated_fat_g": 1.5, "sodium_mg": 210.0, "proteins_g": 7.8, "fiber_g": 6.2, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Whole grain wheat, sugar, cocoa powder, glucose syrup, palm oil, emulsifier (INS 322), artificial flavour vanillin."},

        # Instant Noodles & Convenience Foods
        {"brand": "Maggi", "product_name": "2-Minute Masala Flavoured Noodles", "category": "Instant Noodles", "energy_kcal": 420.0, "sugars_g": 2.2, "fat_g": 15.5, "saturated_fat_g": 7.8, "sodium_mg": 920.0, "proteins_g": 8.5, "fiber_g": 2.8, "fruits_veg_pct": 3.0, "is_beverage": 0, "ingredients_text": "Wheat flour, palm oil, iodized salt, spice mix (coriander, cumin, turmeric, chili), flavor enhancer (INS 621, INS 635), acidity regulator (INS 501i, INS 500i)."},
        {"brand": "Prima", "product_name": "Special Kottu Mee Hot & Spicy", "category": "Instant Noodles", "energy_kcal": 465.0, "sugars_g": 3.5, "fat_g": 20.0, "saturated_fat_g": 9.8, "sodium_mg": 1150.0, "proteins_g": 9.0, "fiber_g": 2.2, "fruits_veg_pct": 2.0, "is_beverage": 0, "ingredients_text": "Wheat flour, palm olein, salt, chili paste, onion, monosodium glutamate (INS 621), disodium 5'-ribonucleotides (INS 635), caramel colour (INS 150c), preservative (INS 211)."},
        {"brand": "Raigam", "product_name": "Deveni Batha Rice Noodles Chicken Curry", "category": "Instant Noodles", "energy_kcal": 360.0, "sugars_g": 1.5, "fat_g": 5.0, "saturated_fat_g": 2.0, "sodium_mg": 840.0, "proteins_g": 7.5, "fiber_g": 2.5, "fruits_veg_pct": 4.0, "is_beverage": 0, "ingredients_text": "Rice flour, wheat starch, salt, spices (curry powder, coriander, ginger, garlic), yeast extract, flavour enhancer (INS 627, INS 631)."},

        # Beverages & Carbonated Drinks
        {"brand": "Elephant House", "product_name": "Ginger Beer (EGB)", "category": "Beverages", "energy_kcal": 52.0, "sugars_g": 12.8, "fat_g": 0.0, "saturated_fat_g": 0.0, "sodium_mg": 15.0, "proteins_g": 0.0, "fiber_g": 0.0, "fruits_veg_pct": 2.0, "is_beverage": 1, "ingredients_text": "Carbonated water, sucrose, natural ginger extract, citric acid (INS 330), preservative (INS 211 sodium benzoate)."},
        {"brand": "Elephant House", "product_name": "Cream Soda Carbonated Soft Drink", "category": "Beverages", "energy_kcal": 56.0, "sugars_g": 13.9, "fat_g": 0.0, "saturated_fat_g": 0.0, "sodium_mg": 20.0, "proteins_g": 0.0, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 1, "ingredients_text": "Carbonated water, cane sugar, permitted food colour (INS 102 tartrazine), acidity regulator (INS 330), artificial cream soda flavour, preservative (INS 211)."},
        {"brand": "Elephant House", "product_name": "Nelli Crush Sweetened Cordial", "category": "Beverages", "energy_kcal": 68.0, "sugars_g": 16.5, "fat_g": 0.0, "saturated_fat_g": 0.0, "sodium_mg": 28.0, "proteins_g": 0.2, "fiber_g": 0.3, "fruits_veg_pct": 25.0, "is_beverage": 1, "ingredients_text": "Water, sugar, nelli fruit pulp (amla), stabilizer (INS 440), citric acid (INS 330), preservative (INS 224 potassium metabisulphite)."},
        {"brand": "Milo", "product_name": "Malted Choco Ready-to-Drink RTD Pack", "category": "Beverages", "energy_kcal": 76.0, "sugars_g": 10.2, "fat_g": 2.1, "saturated_fat_g": 1.2, "sodium_mg": 52.0, "proteins_g": 2.4, "fiber_g": 0.8, "fruits_veg_pct": 0.0, "is_beverage": 1, "ingredients_text": "Water, skimmed milk powder, malt extract (barley), sugar, cocoa powder, palm olein, minerals (calcium phosphate, ferric pyrophosphate), vitamins (B2, B3, B6, B12, C, D)."},
        {"brand": "Coca-Cola", "product_name": "Zero Sugar No Calorie Carbonated Soda", "category": "Beverages", "energy_kcal": 0.3, "sugars_g": 0.0, "fat_g": 0.0, "saturated_fat_g": 0.0, "sodium_mg": 12.0, "proteins_g": 0.0, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 1, "ingredients_text": "Carbonated water, caramel colour (INS 150d), acidity regulators (INS 338, INS 331), artificial sweeteners (INS 951 aspartame, INS 950 acesulfame potassium), caffeine, preservative (INS 211)."},

        # Spreads, Sauces & Condiments
        {"brand": "Kist", "product_name": "Mixed Fruit Jam", "category": "Spreads & Sauces", "energy_kcal": 280.0, "sugars_g": 62.0, "fat_g": 0.1, "saturated_fat_g": 0.0, "sodium_mg": 45.0, "proteins_g": 0.4, "fiber_g": 1.1, "fruits_veg_pct": 45.0, "is_beverage": 0, "ingredients_text": "Sugar, fruit pulp (papaya, pineapple, mango, passion fruit), pectin (INS 440), citric acid (INS 330), permitted colour (INS 122 carmoisine), preservative (INS 211)."},
        {"brand": "Edinborough", "product_name": "Premium Tomato Sauce Bottled", "category": "Spreads & Sauces", "energy_kcal": 130.0, "sugars_g": 24.5, "fat_g": 0.2, "saturated_fat_g": 0.0, "sodium_mg": 890.0, "proteins_g": 1.2, "fiber_g": 1.5, "fruits_veg_pct": 40.0, "is_beverage": 0, "ingredients_text": "Tomato paste, sugar, vinegar, modified corn starch (INS 1422), salt, onion powder, garlic, spices, preservative (INS 211 sodium benzoate, INS 202 potassium sorbate)."},
        {"brand": "MD", "product_name": "Woodapple Nectar Ready to Drink", "category": "Beverages", "energy_kcal": 72.0, "sugars_g": 17.0, "fat_g": 0.1, "saturated_fat_g": 0.0, "sodium_mg": 32.0, "proteins_g": 0.5, "fiber_g": 1.2, "fruits_veg_pct": 35.0, "is_beverage": 1, "ingredients_text": "Water, woodapple pulp, sucrose, pectin (INS 440), citric acid (INS 330), preservative (INS 211)."},
        {"brand": "Marmite", "product_name": "Yeast Extract Savoury Spread", "category": "Spreads & Sauces", "energy_kcal": 260.0, "sugars_g": 1.0, "fat_g": 0.5, "saturated_fat_g": 0.1, "sodium_mg": 4200.0, "proteins_g": 34.0, "fiber_g": 3.0, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Yeast extract (contains barley, wheat, rye), salt, carrot extract, onion extract, spice extracts, niacin, thiamin, riboflavin, folic acid, vitamin B12."},

        # Traditional Snacks & Savouries
        {"brand": "Harischandra", "product_name": "Roasted Kurakkan (Finger Millet) Flour", "category": "Flours & Staples", "energy_kcal": 340.0, "sugars_g": 0.6, "fat_g": 1.3, "saturated_fat_g": 0.3, "sodium_mg": 11.0, "proteins_g": 7.3, "fiber_g": 11.5, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "100% whole grain roasted finger millet (kurakkan)."},
        {"brand": "Keells", "product_name": "Crispy Cassava Chips Chili Salt", "category": "Snacks", "energy_kcal": 520.0, "sugars_g": 2.5, "fat_g": 32.0, "saturated_fat_g": 14.5, "sodium_mg": 680.0, "proteins_g": 2.5, "fiber_g": 4.1, "fruits_veg_pct": 60.0, "is_beverage": 0, "ingredients_text": "Fresh cassava roots, palm olein oil, iodized salt, red chili powder, acidity regulator (INS 330)."},
        {"brand": "Cargills", "product_name": "Spicy Murukku Savoury Mix", "category": "Snacks", "energy_kcal": 495.0, "sugars_g": 1.8, "fat_g": 26.5, "saturated_fat_g": 12.0, "sodium_mg": 820.0, "proteins_g": 9.2, "fiber_g": 3.5, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Gram flour, rice flour, vegetable oil (palm oil), cumin seeds, sesame seeds, red chili powder, salt, leavening agent (INS 500ii)."},
        {"brand": "Dilmah", "product_name": "Pure Ceylon Green Tea with Jasmine", "category": "Beverages", "energy_kcal": 1.0, "sugars_g": 0.0, "fat_g": 0.0, "saturated_fat_g": 0.0, "sodium_mg": 2.0, "proteins_g": 0.1, "fiber_g": 0.0, "fruits_veg_pct": 0.0, "is_beverage": 1, "ingredients_text": "100% pure Ceylon green tea leaves, natural jasmine flowers."},
        {"brand": "Munchee", "product_name": "Kome Crispy Rice Crackers Seaweed", "category": "Snacks", "energy_kcal": 415.0, "sugars_g": 5.2, "fat_g": 8.0, "saturated_fat_g": 3.2, "sodium_mg": 460.0, "proteins_g": 6.8, "fiber_g": 2.0, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Rice, palm oil, sugar, soy sauce, seaweed flakes, salt, disodium inosinate (INS 631)."},
        {"brand": "Prima", "product_name": "Sunrise Crusty Sandwich White Bread", "category": "Biscuits & Bakery", "energy_kcal": 260.0, "sugars_g": 3.8, "fat_g": 2.8, "saturated_fat_g": 0.9, "sodium_mg": 470.0, "proteins_g": 8.5, "fiber_g": 2.3, "fruits_veg_pct": 0.0, "is_beverage": 0, "ingredients_text": "Refined wheat flour, potable water, yeast, cane sugar, iodized salt, vegetable fat, flour treatment agent (INS 300 ascorbic acid), preservative (INS 282 calcium propionate)."}
    ]
    
    # Expand dataset through realistic parametric perturbation to generate 1,200 diverse product profiles
    np.random.seed(42)
    expanded_rows = []
    
    categories = ["Dairy", "Biscuits & Bakery", "Cereals & Breakfast", "Instant Noodles", "Beverages", "Spreads & Sauces", "Snacks", "Flours & Staples"]
    
    for i in range(1200):
        base = records[i % len(records)].copy()
        
        # Add realistic sample-level variance
        jitter = np.random.uniform(0.85, 1.15)
        energy_kcal = float(np.clip(base["energy_kcal"] * jitter, 0.0, 750.0))
        sugars_g = float(np.clip(base["sugars_g"] * np.random.uniform(0.8, 1.2), 0.0, 95.0))
        fat_g = float(np.clip(base["fat_g"] * np.random.uniform(0.8, 1.2), 0.0, 85.0))
        saturated_fat_g = float(np.clip(min(base["saturated_fat_g"] * np.random.uniform(0.8, 1.2), fat_g), 0.0, 45.0))
        sodium_mg = float(np.clip(base["sodium_mg"] * np.random.uniform(0.85, 1.15), 0.0, 5000.0))
        proteins_g = float(np.clip(base["proteins_g"] * np.random.uniform(0.85, 1.15), 0.0, 60.0))
        fiber_g = float(np.clip(base["fiber_g"] * np.random.uniform(0.8, 1.2), 0.0, 30.0))
        fruits_veg_pct = float(np.clip(base["fruits_veg_pct"] * np.random.uniform(0.9, 1.1), 0.0, 100.0))
        is_bev = base["is_beverage"]
        
        # Determine health grade
        grade = compute_health_grade(
            energy_kcal=energy_kcal,
            sugars_g=sugars_g,
            saturated_fat_g=saturated_fat_g,
            sodium_mg=sodium_mg,
            proteins_g=proteins_g,
            fiber_g=fiber_g,
            fruits_veg_pct=fruits_veg_pct,
            is_beverage=bool(is_bev)
        )
        
        # Compute additive count from base text with randomized simulated tags
        additives = extract_e_numbers(base["ingredients_text"])
        additives_count = len(additives)
        if np.random.rand() < 0.15:
            additives_count += np.random.choice([1, 2])
        
        row = {
            "item_id": f"LKP-{i+1001:05d}",
            "brand": base["brand"],
            "product_name": f"{base['product_name']} (Batch #{i+1:04d})",
            "category": base["category"],
            "energy_kcal": round(energy_kcal, 1),
            "sugars_g": round(sugars_g, 2),
            "fat_g": round(fat_g, 2),
            "saturated_fat_g": round(saturated_fat_g, 2),
            "sodium_mg": round(sodium_mg, 1),
            "proteins_g": round(proteins_g, 2),
            "fiber_g": round(fiber_g, 2),
            "fruits_veg_pct": round(fruits_veg_pct, 1),
            "is_beverage": is_bev,
            "additives_count": additives_count,
            "ingredients_text": base["ingredients_text"],
            "health_grade": grade
        }
        expanded_rows.append(row)
        
    df = pd.DataFrame(expanded_rows)
    df.to_csv(filepath, index=False)
    print(f"[DataPipeline] Generated and saved {len(df)} records to {filepath}")
    return df


def load_and_preprocess_dataset(
    filepath: str = DATA_FILE_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], Dict[str, Any]]:
    """
    Loads dataset, extracts feature matrix, normalizes numerical attributes,
    and returns stratified train-test splits alongside feature metadata.
    """
    if not os.path.exists(filepath):
        df = generate_local_food_dataset(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Feature columns for CatBoost tabular classifier
    feature_cols = [
        "energy_kcal",
        "sugars_g",
        "fat_g",
        "saturated_fat_g",
        "sodium_mg",
        "proteins_g",
        "fiber_g",
        "fruits_veg_pct",
        "is_beverage",
        "additives_count"
    ]
    
    target_col = "health_grade"
    
    # Clean missing values if any
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    df[target_col] = df[target_col].fillna("C")
    
    # Target encoding: 'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4
    grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    inv_grade_mapping = {v: k for k, v in grade_mapping.items()}
    
    X = df[feature_cols].copy()
    y = df[target_col].map(grade_mapping)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    metadata = {
        "feature_names": feature_cols,
        "grade_mapping": grade_mapping,
        "inv_grade_mapping": inv_grade_mapping,
        "class_names": ['A', 'B', 'C', 'D', 'E'],
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": df[target_col].value_counts().to_dict()
    }
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return X_train, X_test, y_train, y_test, feature_cols, metadata


def fetch_open_food_facts_api(query: str = "biscuit", page_size: int = 20) -> pd.DataFrame:
    """
    Connects to the official Open Food Facts REST API (world.openfoodfacts.org)
    to query live packaged product records, parse nutritional fields, and normalize
    into the NutriLens AI feature schema.
    """
    import requests
    
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1&page_size={page_size}"
    headers = {"User-Agent": "NutriLensAI - MScUniversityResearch/1.0 (academic@nutrilens.ai)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        payload = response.json()
    except Exception as err:
        print(f"[OpenFoodFacts API] Live API endpoint unavailable ({err}). Using curated local regional benchmark.")
        return pd.read_csv(DATA_FILE_PATH).head(page_size)
    
    products = payload.get("products", [])
    records = []
    
    for p in products:
        nutriments = p.get("nutriments", {})
        energy = float(nutriments.get("energy-kcal_100g") or nutriments.get("energy-kcal") or 0.0)
        sugars = float(nutriments.get("sugars_100g") or 0.0)
        fat = float(nutriments.get("fat_100g") or 0.0)
        sat_fat = float(nutriments.get("saturated-fat_100g") or 0.0)
        sodium = float(nutriments.get("sodium_100g") or 0.0) * 1000.0  # g to mg
        proteins = float(nutriments.get("proteins_100g") or 0.0)
        fiber = float(nutriments.get("fiber_100g") or 0.0)
        fruits_veg = float(nutriments.get("fruits-vegetables-nuts-estimate-from-ingredients_100g") or 0.0)
        
        ing_text = str(p.get("ingredients_text") or "")
        detected_e = extract_e_numbers(ing_text)
        additives_cnt = len(detected_e)
        
        is_bev = 1 if "beverage" in str(p.get("categories", "")).lower() else 0
        
        grade = str(p.get("nutriscore_grade") or "").upper()
        if grade not in ['A', 'B', 'C', 'D', 'E']:
            grade = compute_health_grade(
                energy_kcal=energy, sugars_g=sugars, saturated_fat_g=sat_fat,
                sodium_mg=sodium, proteins_g=proteins, fiber_g=fiber,
                fruits_veg_pct=fruits_veg, is_beverage=bool(is_bev)
            )
            
        records.append({
            "item_id": p.get("code") or p.get("id") or "OFF-UNKNOWN",
            "brand": p.get("brands") or "Generic",
            "product_name": p.get("product_name") or "Imported Food Item",
            "category": (p.get("categories") or "General").split(",")[0],
            "energy_kcal": energy,
            "sugars_g": sugars,
            "fat_g": fat,
            "saturated_fat_g": sat_fat,
            "sodium_mg": sodium,
            "proteins_g": proteins,
            "fiber_g": fiber,
            "fruits_veg_pct": fruits_veg,
            "is_beverage": is_bev,
            "additives_count": additives_cnt,
            "ingredients_text": ing_text,
            "health_grade": grade
        })
        
    df_off = pd.DataFrame(records)
    print(f"[OpenFoodFacts API] Successfully retrieved and normalized {len(df_off)} items.")
    return df_off


def load_kaggle_open_food_facts_csv(csv_path: str, max_rows: int = 5000) -> pd.DataFrame:
    """
    Ingests and normalizes the Kaggle 'World Food Facts' mirror dataset CSV
    (https://www.kaggle.com/datasets/openfoodfacts/world-food-facts)
    mapping legacy column headers into NutriLens AI standard feature matrix.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Kaggle CSV not found at {csv_path}")
        
    df_raw = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    
    col_map = {
        "energy_100g": "energy_kcal",
        "sugars_100g": "sugars_g",
        "fat_100g": "fat_g",
        "saturated-fat_100g": "saturated_fat_g",
        "proteins_100g": "proteins_g",
        "fiber_100g": "fiber_g",
        "nutrition_grade_fr": "health_grade",
        "ingredients_text": "ingredients_text"
    }
    
    df_mapped = pd.DataFrame()
    for raw_col, target_col in col_map.items():
        if raw_col in df_raw.columns:
            df_mapped[target_col] = df_raw[raw_col]
        else:
            df_mapped[target_col] = 0.0
            
    # Normalize sodium (Kaggle sodium is often in grams, convert to mg)
    if "sodium_100g" in df_raw.columns:
        df_mapped["sodium_mg"] = df_raw["sodium_100g"].fillna(0.0) * 1000.0
    else:
        df_mapped["sodium_mg"] = 0.0
        
    df_mapped["fruits_veg_pct"] = df_raw.get("fruits-vegetables-nuts_100g", 0.0).fillna(0.0)
    df_mapped["is_beverage"] = df_raw.get("main_category", "").astype(str).str.contains("beverage", case=False).astype(int)
    
    # Clean grades
    df_mapped["health_grade"] = df_mapped["health_grade"].astype(str).str.upper()
    df_mapped = df_mapped[df_mapped["health_grade"].isin(['A', 'B', 'C', 'D', 'E'])]
    
    print(f"[Kaggle Mirror] Ingested and mapped {len(df_mapped)} records from Kaggle CSV.")
    return df_mapped


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, meta = load_and_preprocess_dataset()
    print("Dataset ingestion and preprocessing successfully verified.")
    print("Feature dimensions:", X_tr.shape, X_te.shape)
    print("Class distribution:", meta["class_distribution"])
