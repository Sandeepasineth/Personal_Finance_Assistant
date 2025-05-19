import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# --- Rule-Based Advisor as Fallback ---
class RuleBasedAdvisor:
    """Provides rule-based financial advice when the AI model is unavailable"""
    
    def get_general_advice(self) -> str:
        """Return general financial advice"""
        return """
## General Financial Advice
- Create and stick to a monthly budget
- Save at least 20% of your income if possible
- Build an emergency fund covering 3-6 months of expenses
- Pay down high-interest debt first
- Invest for long-term goals
- Review your financial plan quarterly
"""
    
    def get_saving_advice(self, income: float, expenses: Dict) -> str:
        """Generate saving advice based on income and expenses"""
        total_expenses = sum(expenses.values()) if expenses else 0
        saving_rate = (income - total_expenses) / income if income > 0 else 0
        
        advice = "## Saving Recommendations\n\n"
        
        if saving_rate < 0:
            advice += """
You're currently spending more than your income. Here are some recommendations:
- Review your expenses to identify non-essential spending that can be reduced
- Consider ways to increase your income through side gigs or asking for a raise
- Create a strict budget to get back to positive savings
- Focus on paying down high-interest debt
- Consider temporary lifestyle adjustments to reduce expenses
"""
        elif saving_rate < 0.1:
            advice += """
Your current saving rate is below 10%, which is lower than recommended. Consider these steps:
- Aim to save at least 15-20% of your income
- Review your largest expense categories for potential savings
- Automate your savings by setting up transfers to a separate account
- Look for opportunities to increase your income
- Create a detailed budget and track your spending closely
"""
        elif saving_rate < 0.2:
            advice += """
You're saving between 10-20% of your income, which is good but could be improved:
- Consider increasing your saving rate to 20-25% if possible
- Make sure you have an emergency fund covering 3-6 months of expenses
- Start investing for retirement if you haven't already
- Review your tax situation for potential savings
- Consider setting specific financial goals to stay motivated
"""
        else:
            advice += """
You're saving 20% or more of your income, which is excellent! Here's how to optimize further:
- Make sure your savings are properly allocated across emergency fund, retirement, and other goals
- Consider investing more aggressively for long-term goals
- Review your investment strategy periodically
- Look into tax-advantaged accounts to maximize your savings
- Consider ways to diversify your income streams
"""
        
        return advice
    
    def get_expense_advice(self, expenses: Dict, income: float) -> str:
        """Generate advice based on expense patterns"""
        if not expenses:
            return "No expense data available to provide specific advice."
            
        # Calculate percentage of income for each category
        total_expenses = sum(expenses.values())
        expense_percentages = {category: amount/income*100 if income > 0 else 0 
                              for category, amount in expenses.items()}
        
        # Sort categories by amount
        sorted_expenses = sorted(expenses.items(), key=lambda x: x[1], reverse=True)
        
        advice = "## Expense Analysis\n\n"
        advice += f"Your top expenses are:\n"
        
        # List top 3 expenses
        for i, (category, amount) in enumerate(sorted_expenses[:3], 1):
            percentage = expense_percentages[category]
            advice += f"{i}. {category}: ${amount:.2f} ({percentage:.1f}% of income)\n"
        
        # Check for potential issues in specific categories
        housing_percent = expense_percentages.get("Housing", 0)
        food_percent = expense_percentages.get("Food", 0)
        entertainment_percent = expense_percentages.get("Entertainment", 0)
        
        advice += "\n### Recommendations:\n"
        
        if housing_percent > 30:
            advice += "- Your housing costs exceed 30% of your income, which is higher than recommended. Consider roommates, negotiating rent, or finding a more affordable option if possible.\n"
            
        if food_percent > 15:
            advice += "- Your food expenses are higher than average. Consider meal planning, cooking at home more often, and reducing dining out.\n"
            
        if entertainment_percent > 10:
            advice += "- Your entertainment spending is relatively high. Look for free or lower-cost alternatives for leisure activities.\n"
            
        if total_expenses > income:
            advice += "- **Warning:** You're spending more than you earn. Identify areas to cut back immediately and consider increasing your income.\n"
            
        advice += "\nGeneral guideline for budget allocation:\n"
        advice += "- Housing: 25-35%\n"
        advice += "- Transportation: 10-15%\n"
        advice += "- Food: 10-15%\n"
        advice += "- Utilities: 5-10%\n"
        advice += "- Healthcare: 5-10%\n"
        advice += "- Savings: 15-25%\n"
        advice += "- Discretionary: 10-20%\n"
        
        return advice
    
    def get_goal_advice(self, goals: List[Dict], income: float, expenses: Dict) -> str:
        """Generate advice based on financial goals"""
        if not goals:
            return """
## Goal Setting Advice
You haven't set any financial goals yet. Consider setting SMART goals:
- Specific: Clearly define what you want to achieve
- Measurable: Set concrete numbers and deadlines
- Achievable: Make sure it's realistic given your situation
- Relevant: Align with your values and long-term objectives
- Time-bound: Set a deadline

Common financial goals to consider:
1. Emergency fund (3-6 months of expenses)
2. Debt reduction or elimination
3. Down payment for a home
4. Retirement savings
5. Education fund
6. Major purchase (car, vacation, etc.)
"""
        
        # Calculate monthly savings capacity
        total_expenses = sum(expenses.values()) if expenses else 0
        monthly_savings_capacity = income - total_expenses
        
        advice = "## Goal Analysis\n\n"
        
        # Analyze each goal
        for goal in goals:
            goal_name = goal['name']
            target_amount = float(goal['target_amount'])
            deadline_str = goal['deadline']
            
            try:
                # Parse deadline
                deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
                today = datetime.now().date()
                
                # Calculate months remaining
                months_remaining = (deadline.year - today.year) * 12 + deadline.month - today.month
                
                # Calculate required monthly saving
                required_monthly_saving = target_amount / months_remaining if months_remaining > 0 else float('inf')
                
                advice += f"### Goal: {goal_name}\n"
                advice += f"- Target amount: ${target_amount:.2f}\n"
                advice += f"- Deadline: {deadline_str}\n"
                advice += f"- Months remaining: {months_remaining}\n"
                advice += f"- Required monthly saving: ${required_monthly_saving:.2f}\n"
                
                # Check feasibility
                if monthly_savings_capacity <= 0:
                    advice += "- **Warning:** You don't currently have capacity to save for this goal. You need to reduce expenses or increase income.\n"
                elif required_monthly_saving > monthly_savings_capacity:
                    advice += f"- **Caution:** This goal requires more than your current monthly savings capacity (${monthly_savings_capacity:.2f}). Consider extending the deadline, reducing the target amount, or increasing your savings capacity.\n"
                else:
                    percentage_of_capacity = required_monthly_saving / monthly_savings_capacity * 100
                    advice += f"- This goal will use {percentage_of_capacity:.1f}% of your current savings capacity.\n"
                
                advice += "\n"
                
            except (ValueError, TypeError):
                advice += f"### Goal: {goal_name}\n"
                advice += "- Unable to analyze this goal due to invalid date format.\n\n"
        
        # Overall goal advice
        if len(goals) > 1:
            total_monthly_requirement = sum(float(goal['target_amount']) / max(1, (datetime.strptime(goal['deadline'], "%Y-%m-%d").date().year - datetime.now().date().year) * 12 + 
                                             datetime.strptime(goal['deadline'], "%Y-%m-%d").date().month - datetime.now().date().month)
                                        for goal in goals if datetime.strptime(goal['deadline'], "%Y-%m-%d").date() > datetime.now().date())
            
            if total_monthly_requirement > monthly_savings_capacity and monthly_savings_capacity > 0:
                advice += "### Overall Goal Assessment\n"
                advice += f"Your goals collectively require ${total_monthly_requirement:.2f} monthly, but your current savings capacity is ${monthly_savings_capacity:.2f}.\n"
                advice += "Consider prioritizing your goals or adjusting their timelines to make them more achievable.\n"
        
        return advice
    
    def get_advice_for_query(self, query: str, user_data: Dict) -> str:
        """Provide rule-based advice based on the query"""
        query_lower = query.lower()
        income = user_data.get('income', 0)
        expenses = self.analyze_expenses_from_data(user_data.get('expenses', []))
        goals = user_data.get('goals', [])
        
        # Match query to advice types
        if any(word in query_lower for word in ['save', 'saving', 'savings']):
            return self.get_saving_advice(income, expenses)
        elif any(word in query_lower for word in ['spend', 'spending', 'expense', 'expenses', 'budget']):
            return self.get_expense_advice(expenses, income)
        elif any(word in query_lower for word in ['goal', 'goals', 'plan', 'target']):
            return self.get_goal_advice(goals, income, expenses)
        else:
            # If no specific match, provide general advice
            general = self.get_general_advice()
            expense = self.get_expense_advice(expenses, income)
            return f"{general}\n\n{expense}"
    
    def analyze_expenses_from_data(self, expenses: List[Dict]) -> Dict:
        """Convert expense list to category:amount dictionary"""
        if not expenses:
            return {}
            
        result = {}
        for expense in expenses:
            category = expense.get('category', 'Other')
            amount = float(expense.get('amount', 0))
            result[category] = result.get(category, 0) + amount
            
        return result

# --- Memory Management ---
class Memory:
    def __init__(self, file_path="user_data.json"):
        self.file_path = file_path
        self.data = self.load_data()
    
    def load_data(self) -> Dict:
        """Load user data from JSON file or create default structure if file doesn't exist"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading data: {str(e)}")
            # If file is corrupted or can't be read, return default structure
        
        return {"users": {}}
    
    def save_data(self) -> bool:
        """Save data to JSON file with error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.file_path) if os.path.dirname(self.file_path) else '.', exist_ok=True)
            
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f, indent=4, default=str)  # Use default=str for date serialization
            return True
        except (IOError, OSError) as e:
            st.error(f"Error saving data: {str(e)}")
            return False
    
    def add_expense(self, user_id: str, category: str, amount: float, date: str) -> bool:
        """Add an expense entry for a user"""
        if not user_id or not category or amount <= 0:
            return False
            
        if user_id not in self.data["users"]:
            self.data["users"][user_id] = {"expenses": [], "goals": [], "income": 0}
            
        self.data["users"][user_id]["expenses"].append({
            "category": category,
            "amount": float(amount),  # Ensure amount is stored as float
            "date": date
        })
        return self.save_data()
    
    def set_goal(self, user_id: str, goal_name: str, target_amount: float, deadline: str) -> bool:
        """Set a financial goal for a user"""
        if not user_id or not goal_name or target_amount <= 0:
            return False
            
        if user_id not in self.data["users"]:
            self.data["users"][user_id] = {"expenses": [], "goals": [], "income": 0}
            
        # Check if goal with same name exists and update it, or add new goal
        goal_exists = False
        for i, goal in enumerate(self.data["users"][user_id]["goals"]):
            if goal["name"] == goal_name:
                self.data["users"][user_id]["goals"][i] = {
                    "name": goal_name,
                    "target_amount": float(target_amount),
                    "deadline": deadline
                }
                goal_exists = True
                break
                
        if not goal_exists:
            self.data["users"][user_id]["goals"].append({
                "name": goal_name,
                "target_amount": float(target_amount),
                "deadline": deadline
            })
            
        return self.save_data()
    
    def set_income(self, user_id: str, income: float) -> bool:
        """Set monthly income for a user"""
        if not user_id or income < 0:
            return False
            
        if user_id not in self.data["users"]:
            self.data["users"][user_id] = {"expenses": [], "goals": [], "income": 0}
            
        self.data["users"][user_id]["income"] = float(income)
        return self.save_data()
    
    def get_user_data(self, user_id: str) -> Dict:
        """Get all data for a specific user"""
        return self.data["users"].get(user_id, {"expenses": [], "goals": [], "income": 0})
    
    def get_user_ids(self) -> List[str]:
        """Get a list of all user IDs"""
        return list(self.data["users"].keys())

# --- AI Agent ---
class FinanceAgent:
    def __init__(self):
        self.memory = Memory()
        
        # Initialize Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found in environment variables. Please create a .env file with your API key.")
            self.llm_available = False
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.7,
                    max_tokens=500
                )
                self.llm_available = True
                
                # Create prompt template for financial advice
                self.prompt = PromptTemplate(
                    input_variables=["user_data", "query"],
                    template="""You are a personal finance assistant. Based on the user's financial data:
                    
{user_data}

Please answer the query: {query}

Provide concise, actionable advice tailored to their specific financial situation. 
Consider their income, spending patterns, and financial goals when giving recommendations.
"""
                )
                
                # Create chain
                self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
                
            except Exception as e:
                st.error(f"Error initializing AI model: {str(e)}")
                self.llm_available = False
                
        # Create a rule-based advisor as fallback
        self.rule_based_advisor = RuleBasedAdvisor()
    
    def analyze_expenses(self, user_id: str) -> Dict:
        """Analyze expenses by category for a user"""
        user_data = self.memory.get_user_data(user_id)
        expenses = user_data["expenses"]
        
        if not expenses:
            return {}
            
        try:
            df = pd.DataFrame(expenses)
            if not df.empty:
                # Ensure amount column is numeric
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                # Remove any rows with NaN amounts
                df = df.dropna(subset=["amount"])
                # Group by category and sum
                summary = df.groupby("category")["amount"].sum().to_dict()
                return summary
        except Exception as e:
            st.error(f"Error analyzing expenses: {str(e)}")
            
        return {}
    
    def get_monthly_trend(self, user_id: str) -> Optional[pd.DataFrame]:
        """Get monthly expense trends"""
        user_data = self.memory.get_user_data(user_id)
        expenses = user_data["expenses"]
        
        if not expenses:
            return None
            
        try:
            df = pd.DataFrame(expenses)
            if not df.empty:
                # Convert date strings to datetime
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                # Extract month and year
                df["month_year"] = df["date"].dt.strftime("%Y-%m")
                # Ensure amount is numeric
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                # Group by month and sum
                monthly = df.groupby("month_year")["amount"].sum().reset_index()
                monthly = monthly.sort_values("month_year")
                return monthly
        except Exception as e:
            st.error(f"Error generating monthly trend: {str(e)}")
            
        return None
    
    def get_advice(self, user_id: str, query: str) -> str:
        """Get AI-powered financial advice with fallback to rule-based system"""
        if not query:
            return "Please enter a question to get financial advice."
            
        # Get user data regardless of which advisor we'll use
        user_data = self.memory.get_user_data(user_id)
        expense_summary = self.analyze_expenses(user_id)
            
        # Attempt to use Gemini API if available
        if self.llm_available:
            try:
                # Format user data for the AI model
                goals_formatted = []
                for goal in user_data["goals"]:
                    goals_formatted.append(f"{goal['name']}: ${goal['target_amount']} by {goal['deadline']}")
                    
                expenses_formatted = []
                for category, amount in expense_summary.items():
                    expenses_formatted.append(f"{category}: ${amount:.2f}")
                    
                data_str = f"""
Monthly Income: ${user_data['income']:.2f}

Expenses by Category:
{', '.join(expenses_formatted) if expenses_formatted else 'No expenses recorded'}

Financial Goals:
{', '.join(goals_formatted) if goals_formatted else 'No goals set'}
"""
                
                # Get response from LLM
                response = self.chain.run(user_data=data_str, query=query)
                return response
                
            except Exception as e:
                error_message = str(e)
                st.warning(f"Gemini API is unavailable. Falling back to rule-based advice. Error: {error_message}")
                
                # On API error, fallback to rule-based system
                return self.rule_based_advisor.get_advice_for_query(query, user_data)
        else:
            # Use rule-based system if LLM is not available
            return self.rule_based_advisor.get_advice_for_query(query, user_data)

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Personal Finance Assistant",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Personal Finance Assistant")
    st.write("Track expenses, set goals, and get personalized financial advice")
    
    # Initialize the finance agent
    agent = FinanceAgent()
    
    # Initialize session state for active section
    if "active_section" not in st.session_state:
        st.session_state.active_section = "Dashboard"
    
    # User Settings in sidebar
    with st.sidebar:
        st.header("User Settings")
        
        # User ID selection
        user_ids = agent.memory.get_user_ids()
        user_options = ["Select a User or Create New"] + user_ids
        
        selected_option = st.selectbox("Select User Account", user_options, key="user_select")
        
        user_id = None
        if selected_option == "Select a User or Create New":
            new_user_id = st.text_input("Enter New User ID", value="", key="new_user_id")
            if new_user_id:
                user_id = new_user_id
                if new_user_id in user_ids:
                    st.warning("This User ID already exists. Data will be loaded for this user.")
                else:
                    agent.memory.set_income(new_user_id, 0)
                    st.success(f"New user '{new_user_id}' created!")
        elif selected_option in user_ids:
            user_id = selected_option
        
        if not user_id:
            st.info("Please select an existing user or enter a new User ID to continue.")
            return
        
        st.caption("This is a demo app. In a production environment, use secure authentication.")
        
        # Navigation buttons
        if st.button("📊 Dashboard", key="dashboard_button"):
            st.session_state.active_section = "Dashboard"
        if st.button("💸 Add Expense", key="expense_button"):
            st.session_state.active_section = "Add Expense"
        if st.button("🎯 Set Goal", key="goal_button"):
            st.session_state.active_section = "Set Goal"
        if st.button("💬 Get Advice", key="advice_button"):
            st.session_state.active_section = "Get Advice"
    
    # Main content area
    if user_id:
        # Dashboard Section
        if st.session_state.active_section == "Dashboard":
            st.subheader("Financial Overview")
            
            user_data = agent.memory.get_user_data(user_id)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Monthly Income", f"${user_data['income']:.2f}")
                
                new_income = st.number_input("Update Monthly Income", 
                                            min_value=0.0, 
                                            value=float(user_data['income']), 
                                            step=100.0,
                                            key="dashboard_income")
                if st.button("Update Income", key="update_income_button"):
                    if agent.memory.set_income(user_id, new_income):
                        st.success("Income updated!")
                    else:
                        st.error("Failed to update income.")
            
            with col2:
                expense_summary = agent.analyze_expenses(user_id)
                total_expenses = sum(expense_summary.values()) if expense_summary else 0
                st.metric("Total Expenses", f"${total_expenses:.2f}")
                
                savings = user_data['income'] - total_expenses
                st.metric("Monthly Savings", f"${savings:.2f}", 
                         delta=f"{(savings/user_data['income']*100):.1f}% of income" if user_data['income'] > 0 else "")
            
            st.subheader("Expense Breakdown")
            if expense_summary:
                df = pd.DataFrame.from_dict(expense_summary, orient="index", columns=["Total Spent"])
                df = df.sort_values("Total Spent", ascending=False)
                st.bar_chart(df)
            else:
                st.info("No expenses recorded yet. Use the 'Add Expense' section to start tracking.")
            
            st.subheader("Monthly Spending Trend")
            monthly_data = agent.get_monthly_trend(user_id)
            if monthly_data is not None and not monthly_data.empty:
                st.line_chart(monthly_data.set_index("month_year"))
            else:
                st.info("Not enough data to show monthly trends yet.")
                
            st.subheader("Financial Goals")
            if user_data["goals"]:
                for goal in user_data["goals"]:
                    goal_progress = st.progress(0.0)
                    col1, col2, col3 = st.columns(3)
                    
                    progress_value = 0.0
                    if goal["target_amount"] > 0:
                        progress_value = min(1.0, savings / goal["target_amount"])
                    
                    with col1:
                        st.write(f"**{goal['name']}**")
                    with col2:
                        st.write(f"Target: ${goal['target_amount']:.2f}")
                    with col3:
                        st.write(f"Deadline: {goal['deadline']}")
                    
                    goal_progress.progress(progress_value)
            else:
                st.info("No financial goals set yet. Use the 'Set Goal' section to create goals.")
        
        # Add Expense Section
        if st.session_state.active_section == "Add Expense":
            st.subheader("Add New Expense")
            
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.selectbox(
                    "Category", 
                    ["Food", "Housing", "Transport", "Entertainment", "Utilities", "Healthcare", "Shopping", "Education", "Other"],
                    key="expense_category"
                )
                custom_category = st.text_input("Or enter a custom category", "", key="custom_category")
                final_category = custom_category if custom_category else category
                
                amount = st.number_input("Amount ($)", min_value=0.01, step=1.0, format="%.2f", key="expense_amount")
            
            with col2:
                date = st.date_input("Date", value=datetime.today(), key="expense_date")
                notes = st.text_area("Notes (optional)", "", height=100, key="expense_notes")
            
            if st.button("Add Expense", key="add_expense_button"):
                if amount <= 0:
                    st.error("Please enter a valid amount greater than zero.")
                else:
                    if agent.memory.add_expense(user_id, final_category, amount, str(date)):
                        st.success(f"✅ ${amount:.2f} expense added to {final_category}!")
                    else:
                        st.error("Failed to add expense. Please check your inputs.")
            
            st.subheader("Recent Expenses")
            user_data = agent.memory.get_user_data(user_id)
            expenses = user_data["expenses"]
            
            if expenses:
                expense_df = pd.DataFrame(expenses)
                expense_df["date"] = pd.to_datetime(expense_df["date"], errors="coerce")
                expense_df = expense_df.sort_values("date", ascending=False).head(5)
                
                expense_df["amount"] = expense_df["amount"].apply(lambda x: f"${float(x):.2f}")
                expense_df["date"] = expense_df["date"].dt.strftime("%Y-%m-%d")
                
                st.dataframe(expense_df[["date", "category", "amount"]], use_container_width=True)
            else:
                st.info("No expenses recorded yet.")
        
        # Set Goal Section
        if st.session_state.active_section == "Set Goal":
            st.subheader("Set Financial Goal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                goal_name = st.text_input("Goal Name (e.g., Save for Vacation)", key="goal_name")
                target_amount = st.number_input("Target Amount ($)", min_value=1.0, step=100.0, format="%.2f", key="target_amount")
            
            with col2:
                deadline = st.date_input("Deadline", value=datetime.today(), key="deadline")
                goal_notes = st.text_area("Goal Notes (optional)", "", height=100, key="goal_notes")
            
            if st.button("Set Goal", key="set_goal_button"):
                if not goal_name:
                    st.error("Please enter a goal name.")
                elif target_amount <= 0:
                    st.error("Please enter a valid target amount greater than zero.")
                else:
                    if agent.memory.set_goal(user_id, goal_name, target_amount, str(deadline)):
                        st.success(f"✅ Goal '{goal_name}' set for ${target_amount:.2f}!")
                    else:
                        st.error("Failed to set goal. Please check your inputs.")
            
            st.subheader("Your Financial Goals")
            user_data = agent.memory.get_user_data(user_id)
            goals = user_data["goals"]
            
            if goals:
                for i, goal in enumerate(goals):
                    with st.expander(f"{goal['name']} - ${float(goal['target_amount']):.2f}"):
                        st.write(f"**Target Amount:** ${float(goal['target_amount']):.2f}")
                        st.write(f"**Deadline:** {goal['deadline']}")
            else:
                st.info("No financial goals set yet.")
        
        # Get Advice Section
        if st.session_state.active_section == "Get Advice":
            st.subheader("Get Personalized Financial Advice")
            
            with st.expander("About the Advice System"):
                st.write("""
                This finance assistant offers two types of advice:
                
                1. **AI-Powered Advice**: Personalized recommendations from Google's Gemini API based on your specific financial situation.
                
                2. **Rule-Based Advice**: If the Gemini API is unavailable (due to quota limits, rate limits, or other issues), 
                the system will automatically fall back to this built-in advisor that provides recommendations based on financial best practices.
                
                Both systems analyze your income, expenses, and goals to provide relevant advice.
                """)
            
            st.write("**Example questions:**")
            example_questions = [
                "How can I save more money?",
                "Am I spending too much on entertainment?",
                "How should I prioritize my financial goals?",
                "What's a good budgeting strategy for me?",
                "How can I reduce my monthly expenses?"
            ]
            
            col1, col2 = st.columns(2)
            with col1:
                for i, question in enumerate(example_questions[:3]):
                    if st.button(question, key=f"example_q_{i}"):
                        st.session_state.advice_query = question
            with col2:
                for i, question in enumerate(example_questions[3:], start=3):
                    if st.button(question, key=f"example_q_{i}"):
                        st.session_state.advice_query = question
            
            if "advice_query" not in st.session_state:
                st.session_state.advice_query = ""
                
            query = st.text_area(
                "Ask for financial advice:", 
                value=st.session_state.advice_query,
                height=100,
                key="advice_text_area"
            )
            
            if st.button("Get Advice", key="get_advice_button"):
                if query:
                    with st.spinner("Generating advice..."):
                        try:
                            advice = agent.get_advice(user_id, query)
                            st.session_state.last_advice = advice
                            st.session_state.last_query = query
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")
                            st.session_state.last_advice = "An unexpected error occurred. Please try again."
                            st.session_state.last_query = query
            
            if "last_advice" in st.session_state:
                st.write("---")
                st.write(f"**Your question:** {st.session_state.last_query}")
                st.write("**Advice:**")
                st.markdown(st.session_state.last_advice)

if __name__ == "__main__":
    main()