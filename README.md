# **Bridging the Explainability Gap In Digital Marketing: A Structural Equation Model for Transparent Media Mix Modeling**

This repository contains the research paper, modeling code, and methodology for **"Bridging the Explainability Gap In Digital Marketing: A Structural Equation Model for Transparent Media Mix Modeling."** This research demonstrates how to shift from opaque, purely predictive Media Mix Modeling (MMM) towards a transparent, causally-driven macroscopic recommendation engine using Covariance-Based Structural Equation Modeling (CB-SEM).

## **📖 Abstract**

The rapid evolution of digital marketing has caused a high level of competition, compelling marketing companies to rely heavily on Media Mix Modeling (MMM) to optimize their budgets. However, traditional MMM models often lack the causal transparency required for strategic decision-making.  
This study bridges this explainability gap by analyzing the interdependent causal mechanisms between marketing channels and business performance, while measuring how managerial response to environmental volatility affects revenue. Using Structural Equation Modeling (SEM) on 135 weeks of proprietary US market data from a leading global consumer electronics manufacturer, the theory-driven structural model ($R^2=0.82$) reveals the causal relations among factors in different levels of the marketing funnel that traditional predictive methods fail to identify. The model demonstrates high robustness and avoids over-fitting, offering a transparent, causal approach to evaluating marketing interventions.

## **🚀 Key Contributions**

1. **Causal Transparency over Pure Prediction:** Overcomes the "Accuracy Paradox" seen in standard out-of-the-box Ridge Regression models, properly valuing the indirect impact of upper-funnel media (like Meta and Video) on high-intent lower-funnel channels (like Branded Search).  
2. **Quantifying Managerial Decision Friction:** Introduces the concept of *Budget Allocation Instability*—empirically proving that reactive budget shifting by human managers during periods of market volatility acts as a structural drag on revenue.  
3. **Deconstructing the Funnel:** Maps out complex digital marketing behaviors including platform substitution effects (e.g., YouTube cannibalizing concurrent search traffic) and structural attrition (e.g., Display media driving low-intent generic searches).

## **📂 Repository Structure**

├── data/  
│   └── consumer_electronics_data.xlsx         
├── paper/  
│   └── Transparent_MMM_RecSys2026.pdf         
├── code/  
│   └── Electronics_Consumer_SEM_US.ipynb        
├── images/  
│   └── Final_Path_Diagram.png                 
├── README.md                                   
└── requirements.txt
## **📊 Dataset & External Controls**

The core dataset consists of 135 weeks of proprietary U.S. retail data from a multinational consumer electronics brand. To protect proprietary information, the raw data is not included. However, the code illustrates the integration of several automated external environmental controls:

* **Macroeconomics:** CPI / Inflation via the Federal Reserve API (fredapi).  
* **Market Uncertainty:** 30-day rolling market volatility (S\&P 500\) via Yahoo Finance (yfinance).  
* **Competitive & Brand Trends:** Relative search interest indexing via Google Trends (trendspy).  
* **Holiday Baseline Adjustments:** Automated federal holiday flagging via the holidays library.

## **⚙️ Installation & Requirements**

Ensure you have Python 3.12+ installed. Install the required dependencies:  
pip install pandas numpy scikit-learn semopy statsmodels fredapi trendspy yfinance holidays matplotlib seaborn plotly

*Note: The code utilizes semopy for Structural Equation Modeling and requires Graphviz installed on your system to generate the causal path diagrams.*

## **💻 Usage**

The primary script is Electronics Consumer \- SEM \- US.py, which is formatted using jupytext (percent format) and can be executed as a standard Python script or converted directly into a Jupyter Notebook.  
To run the pipeline:

1. Ensure your API keys (like the FRED API key) are configured in the script.  
2. Place your raw weekly MMM data in the specified directory.  
3. Execute the script:

python "Electronics Consumer \- SEM \- US.py"

### **Pipeline Overview:**

1. **Data Preprocessing & Funnel Grouping:** Maps raw channel data into intentional groups (e.g., Spend\_Search\_Brand, Spend\_Meta, Spend\_OLV).  
2. **Control Injection:** Fetches and standardizes external stressors (Volatility, CPI, Trends).  
3. **VIF Analysis:** Checks for and resolves multicollinearity.  
4. **SEM Modeling:** Iteratively fits and evaluates structural equations, calculating both direct paths and indirect sequential mediation.  
5. **Cross-Validation:** Uses an expanding-window chronological 70/30 split to evaluate Out-of-Sample predictive power (achieving an out-of-sample MAPE of 8.81%).

## **📈 Model Comparison**

The code includes a benchmark comparison against industry-standard L2-Regularized (Ridge) Regression. The output demonstrates how Ridge Regression algorithmically shrinks highly correlated upper-funnel channels to zero, whereas SEM preserves them by mapping their causal mediation paths.

