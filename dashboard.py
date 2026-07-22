import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="NovaMart Dashboard - Muhammed Anshif K",
    page_icon="📊",
    layout="wide"
)


# -------------------------------------------------
# LOAD AND PREPARE CLEAN DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\dbila\Project_5.1\Project_2_NovaMart\Output\novamart_clean.csv")

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Use canonical lookup columns when merge created _x and _y columns
    if "region_y" in df.columns:
        df["region"] = df["region_y"]
    elif "region" not in df.columns and "region_x" in df.columns:
        df["region"] = df["region_x"]

    if "category_y" in df.columns:
        df["category"] = df["category_y"]
    elif "category" not in df.columns and "category_x" in df.columns:
        df["category"] = df["category_x"]

    if "sub_category_y" in df.columns:
        df["sub_category"] = df["sub_category_y"]
    elif "sub_category" not in df.columns and "sub_category_x" in df.columns:
        df["sub_category"] = df["sub_category_x"]

    # Convert date columns
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(
            df["ship_date"],
            errors="coerce"
        )

    # Convert numeric columns
    numeric_columns = [
        "sales",
        "profit",
        "discount",
        "quantity",
        "profit_margin_pct",
        "shipping_days"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove rows with invalid essential values
    df = df.dropna(
        subset=[
            "order_date",
            "region",
            "category",
            "sales"
        ]
    ).copy()

    # Month fields
    df["month_number"] = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.month_name()

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "The file data/novamart_clean.csv was not found. "
        "Run 01_data_prep.ipynb first."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("📊 NovaMart Dashboard — Muhammed Anshif K")

st.markdown(
    "Regional sales and monthly performance analysis "
    "for NovaMart from January to June 2024."
)


# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

region_options = sorted(
    df["region"].dropna().astype(str).unique().tolist()
)

selected_regions = st.sidebar.multiselect(
    "Select Region",
    options=region_options,
    default=region_options
)

minimum_month = int(df["month_number"].min())
maximum_month = int(df["month_number"].max())

selected_month_range = st.sidebar.slider(
    "Select Month Range",
    min_value=minimum_month,
    max_value=maximum_month,
    value=(minimum_month, maximum_month)
)

filtered_df = df[
    (df["region"].isin(selected_regions))
    & df["month_number"].between(
        selected_month_range[0],
        selected_month_range[1]
    )
].copy()

st.sidebar.markdown("---")
st.sidebar.write(
    f"**Filtered Records:** {len(filtered_df):,}"
)

if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Please change the filter values."
    )
    st.stop()


# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------
total_sales = filtered_df["sales"].sum()
total_profit = filtered_df["profit"].sum()
regions_covered = filtered_df["region"].nunique()

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="Total Sales",
        value=f"${total_sales:,.2f}"
    )

with kpi2:
    st.metric(
        label="Total Profit",
        value=f"${total_profit:,.2f}"
    )

with kpi3:
    st.metric(
        label="Regions Covered",
        value=regions_covered
    )

st.markdown("---")


# -------------------------------------------------
# CHART 1: SALES BY REGION
# -------------------------------------------------
st.subheader("Sales by Region")

regional_sales = (
    filtered_df
    .groupby("region", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
)

sales_region_chart = px.bar(
    regional_sales,
    x="region",
    y="sales",
    text_auto=".2s",
    labels={
        "region": "Region",
        "sales": "Total Sales"
    },
    title="Regional Sales Performance"
)

sales_region_chart.update_layout(
    xaxis_title="Region",
    yaxis_title="Sales",
    showlegend=False
)

st.plotly_chart(
    sales_region_chart,
    use_container_width=True
)


# -------------------------------------------------
# CHART 2: MONTHLY SALES TREND BY REGION
# -------------------------------------------------
st.subheader("Monthly Sales Trend by Region")

monthly_region_sales = (
    filtered_df
    .groupby(
        ["region", "month_number", "month_name"],
        as_index=False
    )["sales"]
    .sum()
    .sort_values(["region", "month_number"])
)

monthly_line_chart = px.line(
    monthly_region_sales,
    x="month_name",
    y="sales",
    color="region",
    markers=True,
    labels={
        "month_name": "Month",
        "sales": "Total Sales",
        "region": "Region"
    },
    title="Monthly Sales Trend for Each Region",
    category_orders={
        "month_name": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June"
        ]
    }
)

monthly_line_chart.update_layout(
    xaxis_title="Month",
    yaxis_title="Sales"
)

st.plotly_chart(
    monthly_line_chart,
    use_container_width=True
)


# -------------------------------------------------
# NUMPY / MONTH-OVER-MONTH REGIONAL ALERT
# -------------------------------------------------
st.subheader("Regional Month-over-Month Growth Alert")

regional_growth = monthly_region_sales.copy()

regional_growth["growth_pct"] = (
    regional_growth
    .groupby("region")["sales"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    * 100
)

declining_rows = regional_growth[
    regional_growth["growth_pct"] < 0
]

if declining_rows.empty:
    st.success(
        "✅ No regional sales decline was found "
        "for the selected period."
    )
else:
    for _, row in declining_rows.iterrows():
        st.warning(
            f"⚠️ {row['region']} region sales dropped by "
            f"{abs(row['growth_pct']):.2f}% in "
            f"{row['month_name']} compared with the previous month."
        )

growth_table = regional_growth[
    [
        "region",
        "month_name",
        "sales",
        "growth_pct"
    ]
].copy()

growth_table.columns = [
    "Region",
    "Month",
    "Sales",
    "Growth %"
]

growth_table["Sales"] = growth_table["Sales"].round(2)
growth_table["Growth %"] = growth_table["Growth %"].round(2)

st.dataframe(
    growth_table,
    use_container_width=True,
    hide_index=True
)


# -------------------------------------------------
# CHART 3: CUMULATIVE SALES AREA CHART
# -------------------------------------------------
st.subheader("Cumulative Sales over the Months")

monthly_total_sales = (
    filtered_df
    .groupby(
        ["month_number", "month_name"],
        as_index=False
    )["sales"]
    .sum()
    .sort_values("month_number")
)

monthly_total_sales["cumulative_sales"] = (
    monthly_total_sales["sales"].cumsum()
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.fill_between(
    monthly_total_sales["month_name"],
    monthly_total_sales["cumulative_sales"],
    alpha=0.5
)

ax.plot(
    monthly_total_sales["month_name"],
    monthly_total_sales["cumulative_sales"],
    marker="o"
)

ax.set_title("Cumulative Monthly Sales")
ax.set_xlabel("Month")
ax.set_ylabel("Cumulative Sales")
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# -------------------------------------------------
# CHART 4: PROFIT BY REGION
# -------------------------------------------------
st.subheader("Profit by Region")

regional_profit = (
    filtered_df
    .groupby("region", as_index=False)["profit"]
    .sum()
    .sort_values("profit", ascending=False)
)

profit_region_chart = px.bar(
    regional_profit,
    x="region",
    y="profit",
    text_auto=".2s",
    labels={
        "region": "Region",
        "profit": "Total Profit"
    },
    title="Regional Profit Performance"
)

profit_region_chart.update_layout(
    xaxis_title="Region",
    yaxis_title="Profit",
    showlegend=False
)

st.plotly_chart(
    profit_region_chart,
    use_container_width=True
)


# -------------------------------------------------
# REGION × CATEGORY PIVOT TABLE
# -------------------------------------------------
st.subheader("Region × Category Sales Pivot Table")

pivot_table = pd.pivot_table(
    filtered_df,
    values="sales",
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Total"
)

st.dataframe(
    pivot_table.style.format("${:,.2f}"),
    use_container_width=True
)


# -------------------------------------------------
# REGIONAL PERFORMANCE SUMMARY
# -------------------------------------------------
st.subheader("Regional Performance Summary")

strongest_region_row = regional_sales.iloc[0]

latest_growth_rows = (
    regional_growth
    .dropna(subset=["growth_pct"])
    .sort_values("month_number")
    .groupby("region", as_index=False)
    .tail(1)
)

st.success(
    f"🏆 Strongest Region by Sales: "
    f"{strongest_region_row['region']} "
    f"with ${strongest_region_row['sales']:,.2f}."
)

if not latest_growth_rows.empty:
    slowing_region = latest_growth_rows.sort_values(
        "growth_pct"
    ).iloc[0]

    if slowing_region["growth_pct"] < 0:
        st.warning(
            f"📉 Region currently slowing down: "
            f"{slowing_region['region']} "
            f"with {slowing_region['growth_pct']:.2f}% "
            f"latest month growth."
        )
    else:
        st.info(
            "No region has negative growth in the latest "
            "selected month."
        )


# -------------------------------------------------
# DOWNLOAD BUTTON
# -------------------------------------------------
st.markdown("---")
st.subheader("Download Filtered Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download NovaMart Filtered CSV",
    data=csv_data,
    file_name="novamart_filtered_data.csv",
    mime="text/csv"
)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")

st.caption(
    "Developed by Muhammed Anshif K | "
    "NovaMart Dashboard | Version 2"
)