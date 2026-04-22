# 🏭 PreciCost — Manufacturing Variance Intelligence

A full analytics solution for manufacturing cost variance tracking, including a **Streamlit web app** and **Power BI dashboard**.

---

## 📁 Folder Structure

```
PreciCost/
│
├── app.py                        ← Streamlit dashboard (run this)
├── requirements.txt              ← Python dependencies
│
├── data/                         ← All CSV source files
│   ├── Actual_Costs.csv
│   ├── Budget_Master.csv
│   ├── Date.csv
│   ├── Machines.csv
│   ├── Production_Logs.csv
│   ├── Rework_Registry.csv
│   └── Shifts.csv
│
└── powerbi/
    ├── PowerBI_Setup_Guide.md    ← Step-by-step Power BI instructions
    ├── PowerBI_M_Queries.txt     ← Power Query (M) code for all tables
    └── PowerBI_DAX_Measures.dax  ← All DAX measures (copy into Power BI)
```

---

## 🚀 Running the Streamlit App

### Prerequisites
- Python 3.9+

### Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📊 Dashboard Features

### Streamlit App (`app.py`)
| Tab | Content |
|---|---|
| 📊 Executive Variance | Gauge chart, variance by stage bar chart, actual vs budget trend |
| ♻️ Waste & Quality | Rework donut chart, treemap by stage/reason, scrap scatter plot |
| 🗺️ Stage Heatmap | Stage×Shift variance heatmap, Machine×Shift production heatmap, summary table |

**Sidebar Filters:** Year, Quarter, Production Line, Stage, Shift

### KPIs Calculated
- **Actual Cost** = Material + Utility + (Payroll Hours × $25)
- **Budgeted Cost** = Units Produced × Budgeted Unit Cost (per stage)
- **Cost Variance %** = (Actual − Budget) / Budget
- **Waste Dollars** = Scrap Units × Avg Unit Cost
- **Rework Leak** = Parts Cost Lost + (Hours Lost × $25)
- **Cost Per Unit** = Actual Cost / Good Units
- **Traffic Light**: 🔴 >10% over | 🟡 0–10% over | 🟢 Under budget

---

## 📋 Power BI Setup

1. Open `powerbi/PowerBI_Setup_Guide.md` — full step-by-step instructions
2. Import CSVs using `powerbi/PowerBI_M_Queries.txt` (paste into Advanced Editor)
3. Create relationships per the guide
4. Add all measures from `powerbi/PowerBI_DAX_Measures.dax`
5. Build 3 pages: Executive Variance, Waste & Quality, Stage Heatmap

---

## 🗄️ Data Schema (Star Schema)

```
                    Dim_Date
                       │
         Dim_Machines ─┼─ Dim_Shifts
                       │
Dim_Budget_Master ─────┤
                       │
            ┌──────────┼──────────┐
            │          │          │
   Fact_    │   Fact_  │  Fact_   │
   Actual_  │   Prod_  │  Rework_ │
   Costs    │   Logs   │  Registry│
```

---

## ⚙️ Configuration

Labor rate is set to **$25/hour** by default.  
To change it, edit line in `app.py`:
```python
LABOR_RATE = 25  # $25/hr
```
