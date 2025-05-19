# Personal Finance Assistant

A Streamlit-based web application designed to help users manage their personal finances by tracking expenses, setting financial goals, and receiving AI-powered financial advice. Built as part of the **Data Science Applications and AI** course assignment for the 5th Semester, 2025, this app uses Google's Gemini API for personalized advice and a rule-based advisor as a fallback, with all monetary values displayed in **Sri Lankan Rupees (LKR)**.

## Features

- **Expense Tracking**: Log expenses by category, amount, and date, with a dashboard showing expense breakdowns and monthly trends.
- **Financial Goals**: Set and track SMART financial goals with target amounts and deadlines.
- **AI-Powered Advice**: Get personalized financial recommendations using Google's Gemini API, tailored to your income, expenses, and goals.
- **Rule-Based Fallback**: A built-in advisor provides financial advice based on best practices when the Gemini API is unavailable.
- **User Management**: Supports multiple users via a User ID dropdown, with data persistence in `user_data.json`.
- **Localized Currency**: All monetary values are displayed in **Sri Lankan Rupees (LKR)** for relevance to Sri Lankan users.
- **Sidebar Navigation**: Intuitive navigation via sidebar buttons (Dashboard, Add Expense, Set Goal, Get Advice) without tabs in the main content area.
- **Data Visualization**: Bar charts for expense categories and line charts for monthly spending trends.

## Prerequisites

- **Python 3.8+**: Ensure Python is installed.
- **Virtual Environment**: Recommended for dependency management.
- **Google Gemini API Key**: Required for AI-powered advice.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/personal-finance-assistant.git
   cd personal-finance-assistant
