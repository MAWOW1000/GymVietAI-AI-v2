import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class DietaryRestriction(Enum):
    NONE = "none"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"

class Allergen(Enum):
    NONE = "none"
    NUTS = "nuts"
    PEANUTS = "peanuts"
    DAIRY = "dairy"
    EGGS = "eggs"
    SOY = "soy"
    FISH = "fish"
    SHELLFISH = "shellfish"
    WHEAT = "wheat"
    SESAME = "sesame"

@dataclass
class MealConstraints:
    """Defines nutritional constraints for a meal"""
    min_protein: float = 20.0
    max_fat: float = 25.0
    min_calories: float = 300.0
    macro_tolerance: float = 0.1  # 10% tolerance

class NutritionPlanner:
    # Define restricted food groups for each dietary restriction
    DIET_RESTRICTIONS = {
        DietaryRestriction.VEGETARIAN: {'excluded_foods': {'beef', 'chicken', 'fish', 'pork', 'lamb', 'turkey'}},
        DietaryRestriction.VEGAN: {'excluded_groups': {'Dairy', 'Protein'}, 
                                  'allowed_proteins': {'tofu', 'tempeh', 'seitan', 'lentils', 'chickpeas', 'beans'}},
        DietaryRestriction.GLUTEN_FREE: {'excluded_foods': {'wheat', 'barley', 'rye', 'oats'}},
        DietaryRestriction.DAIRY_FREE: {'excluded_groups': {'Dairy'}}
    }

    # Define allergen-food mappings
    ALLERGEN_FOODS = {
        Allergen.NUTS: {
            'excluded_foods': {'almonds', 'walnuts',  'peanuts', 'almond butter','cashew butter'},
            'ingredients_contains': ['nut']
        },
        Allergen.PEANUTS: {
            'excluded_foods': {'peanuts', 'peanut butter'}, 
            'ingredients_contains': ['peanut']
        },
        Allergen.DAIRY: {
            'excluded_foods': {'milk', 'cheese', 'yogurt', 'cottage cheese', 'butter', 'cream', 'sour cream', 'ice cream', 'greek yogurt'},
            'ingredients_contains': ['milk', 'dairy', 'lactose', 'whey', 'casein']
        },
        Allergen.EGGS: {
            'excluded_foods': {'eggs'},
            'ingredients_contains': ['egg']
        },
        Allergen.SOY: {
            'excluded_foods': {'tofu'},
            'ingredients_contains': ['soy']
        },
        Allergen.FISH: {
            'excluded_foods': {'salmon', 'tuna', 'cod', 'tilapia', 'sardines', 'mackerel', "fish"},
            'ingredients_contains': ['fish']
        },
        Allergen.SHELLFISH: {
            'excluded_foods': {'shrimp', 'crab', 'lobster', 'mussels', 'clams', 'oysters'},
            'ingredients_contains': ['shellfish', 'crustacean'] 
        },
        Allergen.WHEAT: {
            'excluded_foods': {'wheat', 'whole wheat bread', 'bread'},
            'ingredients_contains': ['wheat', 'gluten']
        }
    }

    def __init__(self, food_data: pd.DataFrame, macro_targets: Dict[str, float], 
                 dietary_restrictions: Set[DietaryRestriction] = None,
                 allergens: Set[Allergen] = None):
        """
        Initialize the nutrition planner with food data and daily macro targets
        
        Args:
            food_data: DataFrame with columns [name, calories, protein, carbs, fat, food_group]
            macro_targets: Dictionary with daily targets for calories, protein, carbs, and fat
            dietary_restrictions: Set of DietaryRestriction enums
            allergens: Set of Allergen enums
        """
        self.food_data = food_data
        self.daily_targets = macro_targets
        self.dietary_restrictions = dietary_restrictions or {DietaryRestriction.NONE}
        self.allergens = allergens or {Allergen.NONE}
        self.meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.35,
            'dinner': 0.30,
            'snack': 0.10
        }
        
        # Apply dietary restrictions and allergen filtering to food data
        self.filtered_food_data = self._apply_dietary_restrictions(food_data)
        self.filtered_food_data = self._apply_allergen_restrictions(self.filtered_food_data)
    
    def _apply_allergen_restrictions(self, food_data: pd.DataFrame) -> pd.DataFrame:
        """Filter food data based on allergen restrictions"""
        if Allergen.NONE in self.allergens:
            return food_data
            
        filtered_data = food_data.copy()
        for allergen in self.allergens:
            if allergen == Allergen.NONE:
                continue
                
            allergen_rules = self.ALLERGEN_FOODS[allergen]
            
            # Exclude specific foods
            if 'excluded_foods' in allergen_rules:
                filtered_data = filtered_data[~filtered_data['name'].str.lower().isin(
                    [f.lower() for f in allergen_rules['excluded_foods']]
                )]
            
            # Exclude foods containing allergen ingredients
            if 'ingredients_contains' in allergen_rules:
                for ingredient in allergen_rules['ingredients_contains']:
                    filtered_data = filtered_data[~filtered_data['name'].str.lower().str.contains(ingredient)]
        
        return filtered_data

    def _apply_dietary_restrictions(self, food_data: pd.DataFrame) -> pd.DataFrame:
        """Filter food data based on dietary restrictions"""
        if DietaryRestriction.NONE in self.dietary_restrictions:
            return food_data
        
        filtered_data = food_data.copy()
        for restriction in self.dietary_restrictions:
            if restriction == DietaryRestriction.NONE:
                continue
                
            restriction_rules = self.DIET_RESTRICTIONS[restriction]
            
            # Exclude specific foods
            if 'excluded_foods' in restriction_rules:
                filtered_data = filtered_data[~filtered_data['name'].isin(restriction_rules['excluded_foods'])]
            
            # Exclude food groups
            if 'excluded_groups' in restriction_rules:
                filtered_data = filtered_data[~filtered_data['food_group'].isin(restriction_rules['excluded_groups'])]
                
            # Allow specific foods from excluded groups
            if 'allowed_proteins' in restriction_rules:
                allowed_proteins = restriction_rules['allowed_proteins']
                protein_foods = food_data[food_data['name'].isin(allowed_proteins)]
                filtered_data = pd.concat([filtered_data, protein_foods]).drop_duplicates()
        
        return filtered_data
    
    
        
    def calculate_meal_targets(self, meal_type: str) -> Dict[str, float]:
        """Calculate macro targets for a specific meal based on daily distribution"""
        ratio = self.meal_distribution[meal_type]
        return {
            'calories': self.daily_targets['calories'] * ratio,
            'protein': self.daily_targets['protein'] * ratio,
            'carbs': self.daily_targets['carbs'] * ratio,
            'fat': self.daily_targets['fat'] * ratio
        }
    
    def calculate_portion_size(self, food: pd.Series, remaining_targets: Dict[str, float]) -> float:
        """
        Calculate optimal portion size based on all macro targets
        """
        portions = []
        target_calories = remaining_targets['calories']
        
        # Calculate portion needed for calories first as primary constraint
        if food['calories'] > 0:
            calorie_portion = (target_calories / food['calories']) * 100
            portions.append(min(calorie_portion, 300))  # Cap at 300g
        
        # Calculate portions needed for other macros
        for macro in ['protein', 'carbs', 'fat']:
            if food[macro] > 0 and remaining_targets[macro] > 0:
                macro_portion = (remaining_targets[macro] / food[macro]) * 100
                portions.append(min(macro_portion, 300))  # Cap at 300g
        
        # Get the balanced portion that satisfies most constraints
        optimal_portion = np.median(portions) if portions else 100
        
        # Apply minimum and maximum constraints
        portion = max(min(optimal_portion, 300), 30)  # Between 30g and 300g
        
        # Adjust portion if it would exceed calorie target significantly
        calorie_contribution = (food['calories'] * portion / 100)
        if calorie_contribution > target_calories * 1.1:  # More than 10% over target
            portion = (target_calories / food['calories']) * 100
        
        return round(portion)
    
    def select_foods_for_meal(self, meal_type: str, constraints: MealConstraints) -> List[Tuple[str, float]]:
        """
        Select foods and their portions for a meal with improved macro adherence
        """
        meal_targets = self.calculate_meal_targets(meal_type)
        selected_foods = []
        remaining_targets = meal_targets.copy()
        
        min_calories = meal_targets['calories'] * 0.95  # 5% lower than target
        max_calories = meal_targets['calories'] * 1.05  # 5% higher than target
        
        meal_priorities = {
            'breakfast': ['Grains', 'Dairy', 'Fruits'],
            'lunch': ['Protein', 'Vegetables', 'Grains'],
            'dinner': ['Protein', 'Vegetables', 'Healthy Fats'],
            'snack': ['Fruits', 'Dairy', 'Healthy Fats']
        }
        
        current_calories = 0
    
        for food_group in meal_priorities[meal_type]:
            if current_calories >= max_calories:
                break
                
            group_foods = self.filtered_food_data[self.filtered_food_data['food_group'] == food_group]
            
            if len(group_foods) == 0:
                continue
            
            # Sort foods by protein content for protein-focused meals
            if food_group == 'Protein':
                group_foods = group_foods.sort_values('protein', ascending=False)
            
            # Select food with best macro ratio
            food = group_foods.sample(1).iloc[0]
            
            # Calculate portion based on remaining targets
            portion = self.calculate_portion_size(food, remaining_targets)
            
            # Check if adding this food would exceed max calories
            food_calories = food['calories'] * portion / 100
            if current_calories + food_calories > max_calories:
                # Adjust portion to fit within calorie limit
                remaining_calories = max_calories - current_calories
                portion = (remaining_calories / food['calories']) * 100
            
            if portion > 0:
                selected_foods.append((food['name'], portion))
                current_calories += food['calories'] * portion / 100
                
                # Update remaining targets
                for macro in remaining_targets:
                    remaining_targets[macro] -= (food[macro] * portion / 100)
                    remaining_targets[macro] = max(0, remaining_targets[macro])
        
        # If we're under minimum calories, adjust portions up
        if current_calories < min_calories and selected_foods:
            scale_factor = min_calories / current_calories
            selected_foods = [(name, portion * scale_factor) for name, portion in selected_foods]
        
        return selected_foods
    
    def generate_daily_plan(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate a daily meal plan with improved calorie consistency
        """
        max_attempts = 5
        best_plan = None
        best_deviation = float('inf')
        target_calories = self.daily_targets['calories']
        
        for _ in range(max_attempts):
            constraints = MealConstraints()
            plan = {}
            
            for meal_type in self.meal_distribution.keys():
                plan[meal_type] = self.select_foods_for_meal(meal_type, constraints)
            
            # Calculate total calories and deviation
            totals = self.calculate_plan_nutrition(plan)
            calories_deviation = abs(totals['calories'] - target_calories) / target_calories
            macro_deviation = sum(abs(totals[macro] - self.daily_targets[macro]) / self.daily_targets[macro] 
                                for macro in ['protein', 'carbs', 'fat'])
            
            # Weighted deviation score (prioritize calorie accuracy)
            deviation = calories_deviation * 2 + macro_deviation
            
            if deviation < best_deviation:
                best_deviation = deviation
                best_plan = plan
            
            # If we're within 5% of calorie target and 10% of macro targets, accept this plan
            if calories_deviation < 0.05 and macro_deviation < 0.3:
                break
        
        return best_plan
    
    def calculate_plan_nutrition(self, plan: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
        """Calculate total nutrition facts for the meal plan"""
        totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
        
        for meal_foods in plan.values():
            for food_name, grams in meal_foods:
                food_data = self.filtered_food_data[self.filtered_food_data['name'] == food_name].iloc[0]
                for macro in totals.keys():
                    totals[macro] += food_data[macro] * grams / 100
                    
        # Round the totals to 2 decimal places
        return {k: round(v, 2) for k, v in totals.items()}

    def validate_plan(self, plan: Dict[str, List[Tuple[str, float]]]) -> bool:
        """
        Validate the meal plan against dietary guidelines and constraints with stricter tolerances
        """
        totals = self.calculate_plan_nutrition(plan)
        tolerance = 0.1  # 10% tolerance
        
        # Check if each macro is within tolerance range
        for macro, target in self.daily_targets.items():
            if macro == 'protein':
                # Protein should be at least the target amount
                if totals[macro] < target:
                    return False
            else:
                # Other macros should be within tolerance range
                if abs(totals[macro] - target) > (target * tolerance):
                    return False
        
        return True
    
    def generate_weekly_plan(self) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """
        Generate a meal plan for an entire week
        Returns a dictionary with days as keys and meal plans as values
        """
        DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_plan = {}
        
        # Keep track of used foods to ensure variety
        used_foods = set()
        
        for day in DAYS:
            # Create a temporary copy of food data excluding recently used foods
            temp_food_data = self.filtered_food_data[
                ~self.filtered_food_data['name'].isin(used_foods)
            ].copy()
            
            # If we've used too many foods, reset the used foods set
            if len(temp_food_data) < 10:  # Minimum threshold for food options
                used_foods.clear()
                temp_food_data = self.filtered_food_data.copy()
            
            # Store original food data
            original_food_data = self.filtered_food_data
            # Use temporary food data for this day's plan
            self.filtered_food_data = temp_food_data
            
            # Generate plan for the day
            daily_plan = self.generate_daily_plan()
            
            # Validate the daily plan
            if not self.validate_plan(daily_plan):
                # If the plan is not valid, regenerate it
                daily_plan = self.generate_daily_plan()
            
            # Restore original food data
            self.filtered_food_data = original_food_data
            
            # Add foods to used foods set
            for meal_foods in daily_plan.values():
                for food_name, _ in meal_foods:
                    used_foods.add(food_name)
            
            weekly_plan[day] = daily_plan
        
        return weekly_plan
    
    def calculate_weekly_nutrition(self, weekly_plan: Dict[str, Dict[str, List[Tuple[str, float]]]]
                                ) -> Dict[str, Dict[str, float]]:
        """Calculate nutrition totals for each day of the week"""
        weekly_nutrition = {}
        
        for day, daily_plan in weekly_plan.items():
            weekly_nutrition[day] = self.calculate_plan_nutrition(daily_plan)
            
        return weekly_nutrition
    
    def validate_weekly_plan(self, weekly_plan: Dict[str, Dict[str, List[Tuple[str, float]]]]) -> bool:
        """
        Validate the weekly plan to ensure each day's totals are within an acceptable range
        """
        tolerance = 0.1  # 10% tolerance
        for day, daily_plan in weekly_plan.items():
            totals = self.calculate_plan_nutrition(daily_plan)
            for macro, target in self.daily_targets.items():
                if abs(totals[macro] - target) > (target * tolerance):
                    return False
        return True

    def format_weekly_plan(self, weekly_plan: Dict[str, Dict[str, List[Tuple[str, float]]]], 
                          weekly_nutrition: Dict[str, Dict[str, float]]) -> str:
        """Format the weekly plan into a readable string"""
        output = []
        
        for day, daily_plan in weekly_plan.items():
            output.append(f"\n=== {day} ===")
            
            # Add meals
            for meal_type, foods in daily_plan.items():
                output.append(f"\n{meal_type.capitalize()}:")
                for food_name, grams in foods:
                    output.append(f" - {food_name}: {round(grams)}g")
            
            # Add daily nutrition totals
            nutrition = weekly_nutrition[day]
            output.append("\nDaily Totals:")
            output.append(f"Calories: {nutrition['calories']}kcal")
            output.append(f"Protein: {nutrition['protein']}g")
            output.append(f"Carbs: {nutrition['carbs']}g")
            output.append(f"Fat: {nutrition['fat']}g")
            output.append("\n" + "-"*40)
            
        return "\n".join(output)
    
    
# Load food data
df = pd.read_csv('./data/food_dataset_new.csv')

macro_targets = {'calories': 2520, 'protein': 220, 'carbs': 283, 'fat': 56}

# Create planner instance
planner = NutritionPlanner(
    df,
    macro_targets,
    dietary_restrictions={DietaryRestriction.NONE},
    allergens={Allergen.NONE}
)

# Generate weekly plan
weekly_plan = planner.generate_weekly_plan()
weekly_nutrition = planner.calculate_weekly_nutrition(weekly_plan)

# Print formatted plan
print(planner.format_weekly_plan(weekly_plan, weekly_nutrition))