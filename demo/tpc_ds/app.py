import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from pathlib import Path
from trilogy_public_models import get_executor
from trilogy.core.models import DataType
from trilogy import Executor
from dataclasses import dataclass, field
from datetime import timedelta, datetime
from enums import ChartType
from trilogy.core.enums import FunctionClass, FunctionType


def get_trilogy_executor() -> Executor:
    if "executor" not in st.session_state:
        st.session_state["executor"] = get_executor("duckdb.tpc_ds")
    return st.session_state["executor"]


def set_defaults():
    pass


executor: Executor = get_trilogy_executor()

DEFAULT_CHART = ChartType.BAR_CHART
DEFAULT_METRIC = 'item.current_price'
DEFAULT_DIMENSION = 'date.date'

# unique env per query to avoid pollutions
executor.environment = executor.environment.duplicate()

SKIP_NS_LIST = [
    "warehouse",
    "promotion",
    "call_center",
    "local",
    "date",
    "item",
    "store",
    "time",
    "customer",
    "customer_demographic",
]

namespace_list = sorted(
    list(
        set(
            [
                k.namespace
                for k in executor.environment.concepts.values()
                if k.namespace
                and not k.namespace.startswith("__")
                and k.address.count(".") == 1
                and k.namespace not in SKIP_NS_LIST
            ]
        )
    )
)


@dataclass
class QueryInfo:
    trilogy_text: str = ""
    query_text: str = ""
    columns: list[str] = field(default_factory=list)
    duration: timedelta = timedelta(seconds=0)
    parse_duration: timedelta = timedelta(seconds=0)
    render_duration: timedelta = timedelta(seconds=0)


@dataclass
class QueryResults:
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ExceptionInfo:
    text: str = ""


@dataclass
class MetricInput:
    metric: str = ""
    aggregation: FunctionType = FunctionType.SUM


@dataclass
class GlobalConfig:
    view_type: ChartType = ChartType.US_MAP
    filter: str = ""
    query_info: QueryInfo = field(default_factory=QueryInfo)
    query_results: QueryResults = field(default_factory=QueryResults)
    view_metric_count: int = 1
    active_metrics: list[MetricInput] = field(default_factory=list)
    active_dimensions: list[str] = field(default_factory=list)
    query_invalid: bool = False
    exception: ExceptionInfo = field(default_factory=ExceptionInfo)


CONFIG = GlobalConfig()

st.set_page_config(
    page_title="TPC-DS Exploration",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_valid_dimensions(root_namespace: str, active_view: ChartType, optional: bool = False):
    if optional:
        base = [None]
    else:
        base = []
    if active_view == ChartType.US_MAP:
        return base+sorted(
            [
                k.removeprefix(root_namespace + ".")
                for k, concept in executor.environment.concepts.items()
                if concept.datatype in (DataType.STRING,)
                and concept.address.endswith("state")
                and not concept.name.startswith("_")
                and k.startswith(root_namespace + ".")
            ]
        )
    if active_view == ChartType.LINE_CHART:
        return base+sorted(
            [
                k.removeprefix(root_namespace + ".")
                for k, concept in executor.environment.concepts.items()
                if concept.datatype
                in (DataType.DATE, DataType.DATETIME, DataType.INTEGER)
                and not concept.name.startswith("_")
                and k.startswith(root_namespace + ".")
            ]
        )
    if active_view == ChartType.BAR_CHART:
        return base+sorted(
            [
                k.removeprefix(root_namespace + ".")
                for k, concept in executor.environment.concepts.items()
                if concept.datatype
                in (DataType.DATE, DataType.DATETIME, DataType.INTEGER, DataType.STRING)
                and not concept.name.startswith("_")
                and k.startswith(root_namespace + ".")
            ]
        )
    return base+sorted(
        [
            k.removeprefix(root_namespace + ".")
            for k, concept in executor.environment.concepts.items()
            if concept.datatype in (DataType.STRING, DataType.DATE, DataType.DATETIME)
            and not concept.name.startswith("_")
            and k.startswith(root_namespace + ".")
        ]
    )


def get_valid_metrics(root_namespace: str):
    return sorted(
        [
            k.removeprefix(root_namespace + ".")
            for k, concept in executor.environment.concepts.items()
            if concept.datatype in (DataType.BIGINT, DataType.FLOAT)
            and k.startswith(root_namespace + ".")
        ]
    )


def fetch_data():


    metrics = []
    dimensions = []
    val_labels = []
    ordering = []
    for idx, dimension in enumerate(CONFIG.active_dimensions):
        if not dimension:
            continue
        dimensions.append(dimension)
        if idx == 0:
            ordering.append(f"{dimension} asc")
    for metric in CONFIG.active_metrics:
        metrics.append(
            f"{metric.aggregation.value}({metric.metric}) as {metric.metric}_{metric.aggregation.value.lower()}"
        )
        val_labels.append(f"{metric.metric}_{metric.aggregation.value.lower()}")
        ordering.append(f"{metric.metric}_{metric.aggregation.value.lower()} desc")
    metric_string = ",\n\t".join(metrics)
    dimension_string = ",\n\t".join(dimensions)
    base_query = f"""
select 
    {dimension_string},
    {metric_string},
"""
    if len(CONFIG.filter) > 0:
        base_query += f"""
where
    {CONFIG.filter}"""

    order_string = ",\n\t".join(ordering)
    order_by = f"""
order by {order_string}
"""

    base_query += order_by

    base_query += ";"
    parse_start = datetime.now()
    try:
        query_text = executor.generate_sql(base_query)[-1]
    except Exception as e:
        CONFIG.query_invalid = True
        CONFIG.exception.text = str(e)
        return
    CONFIG.query_info.trilogy_text = base_query
    CONFIG.query_info.query_text = query_text
    CONFIG.query_info.parse_duration = datetime.now() - parse_start
    run_start = datetime.now()
    input_df = pd.DataFrame.from_records(
        executor.execute_raw_sql(query_text).fetchall(),
        columns=[
            *dimensions,
            *val_labels,
        ],
    )
    for val_label in val_labels:
        input_df[val_label] = (input_df[val_label]).apply(
            pd.to_numeric, errors="coerce"
        )
    for dim in dimensions:

        if executor.environment.concepts[dim].datatype in (
            DataType.DATE,
            DataType.DATETIME,
        ):
            input_df[dim] = (input_df[dim]).apply(
                pd.to_datetime, errors="coerce"
            )
    CONFIG.query_results.data = input_df
    CONFIG.query_info.duration = datetime.now() - run_start


def get_cached_selector_value(field_name: str, options: list):
    if field_name in st.session_state:
        cached = st.session_state[field_name]
        if cached in options:
            return options.index(cached)
    return 0

def set_cached_value_if_not_exists(field_name: str, value):
    if field_name not in st.session_state:
        st.session_state[field_name] = value

def set_cached_value(field_name: str, value):
    st.session_state[field_name] = value

set_cached_value_if_not_exists("chart_type", DEFAULT_CHART.value)
set_cached_value_if_not_exists("dimension_0", DEFAULT_DIMENSION)
set_cached_value_if_not_exists("dimension_0", None)
set_cached_value_if_not_exists("metric_0", DEFAULT_METRIC)
set_cached_value_if_not_exists("metric_0_agg", FunctionType.SUM.value)

with st.sidebar:
    st.title("TPC-DS Exploration")
    charts = list(ChartType._value2member_map_.keys())
    active_view = st.selectbox(
        "Select Chart Type", 
        charts, 
        index=get_cached_selector_value(f"chart_type", charts),
        key='chart_type'
    )
    # chart seutup
    CONFIG.view_type = ChartType(active_view)
    if CONFIG.view_type == ChartType.US_MAP:
        view_metric_count = 1
        dimension_count = 1
    elif CONFIG.view_type == ChartType.BAR_CHART:
        view_metric_count = 1
        dimension_count = 2
    elif CONFIG.view_type == ChartType.LINE_CHART:
        view_metric_count = 1
        dimension_count = 2
    else:
        view_metric_count = st.number_input("Number of Metrics", min_value=1, value=1)
        dimension_count = st.number_input("Number of Dimensions", min_value=1, value=1)

    # namespace setup
    root_namespace = st.selectbox("Root Namespace", namespace_list, index=0)

    # dimension setup
    dimension_list = get_valid_dimensions(root_namespace, CONFIG.view_type)
    for dimension in range(dimension_count):
        optional = dimension>0
        if optional:
            dimension_list = [None]+dimension_list
        dimension_selection = st.selectbox(
            f"Dimension {dimension}",
            dimension_list,
            index=get_cached_selector_value(f"dimension_{dimension}", dimension_list),
            key=f"dimension_{dimension}",
            placeholder="Select Dimension to Add",
        )
        if str(dimension_selection) != 'None':
            CONFIG.active_dimensions.append(f"{root_namespace}.{dimension_selection}")

    # metric setup
    st.markdown("### Metric Selection")
    metric_list = get_valid_metrics(root_namespace)
    for metric in range(view_metric_count):
        metric_selection = st.selectbox(
            "Metric",
            metric_list,
            index=get_cached_selector_value(f"metric_{metric}", metric_list),
            key=f"metric_{metric}",
            placeholder="Select Metric to Add",
        )
        aggregation_method = st.selectbox(
            "Aggregation",
            [x.value for x in FunctionClass.AGGREGATE_FUNCTIONS.value],
            index=get_cached_selector_value(
                f"metric_{metric}_agg",
                [x.value for x in FunctionClass.AGGREGATE_FUNCTIONS.value],
            ),
            key=f"metric_{metric}_agg",
        )
        if aggregation_method:
            CONFIG.active_metrics.append(
                MetricInput(
                    metric=f"{root_namespace}.{metric_selection}",
                    aggregation=FunctionType(aggregation_method),
                )
            )

    filter = st.text_input("Filter")
    CONFIG.filter = filter

    if CONFIG.active_metrics:
        fetch_data()


def make_choropleth(input_color_theme: str = "viridis"):
    shading = CONFIG.active_metrics[0]
    val_label = f"{shading.metric}_{shading.aggregation.value.lower()}"
    input_df = CONFIG.query_results.data
    state_col = CONFIG.active_dimensions[0]
    render_start = datetime.now()

    choropleth = px.choropleth(
        input_df,
        locations=state_col,
        color=val_label,
        locationmode="USA-states",
        color_continuous_scale=input_color_theme,
        range_color=(0, max(input_df[val_label])),
        scope="usa",
        labels={val_label: state_col},
    )
    choropleth.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )
    CONFIG.query_info.render_duration = datetime.now() - render_start
    return choropleth


def make_table(input_color_theme: str = "viridis"):
    render_start = datetime.now()
    table = st.dataframe(
        data=CONFIG.query_results.data,
        hide_index=True,
        use_container_width=True,
        height=1000,
    )
    CONFIG.query_info.render_duration = datetime.now() - render_start
    return table

def process_data_for_altair()->tuple[pd.DataFrame, list[str], list[str]]:
    dimensions = [x.replace(".", "_") for x in CONFIG.active_dimensions]
    metrics = [f"{metric.metric}_{metric.aggregation.value.lower()}".replace(".", "_") for metric in CONFIG.active_metrics]
    data = CONFIG.query_results.data
    data.columns = [*dimensions, *metrics]
    return data, dimensions, metrics

def make_line_chart():
    render_start = datetime.now()
    data, dimensions, metrics = process_data_for_altair()
    linechart = st.line_chart(
        data=data,
        x=dimensions[0],
        y=metrics[0],
        color = dimensions[1] if len(dimensions)>1 else None,
        use_container_width=True,
        height=500,
    )

    CONFIG.query_info.render_duration = datetime.now() - render_start
    return linechart


def make_bar_chart():
    render_start = datetime.now()
    data, dimensions, metrics = process_data_for_altair()
    linechart = st.bar_chart(
        data=data,
        y=metrics[0],
        y_label=metrics[0],
        x=dimensions[0],
        x_label=dimensions[0],
        color = dimensions[1] if len(dimensions)>1 else None,
        use_container_width=True,
        height=500,
    )
    CONFIG.query_info.render_duration = datetime.now() - render_start
    return linechart

def make_scatter_plot():
    render_start = datetime.now()
    data, dimensions, metrics = process_data_for_altair()
    scatterplot = st.scatter_chart(
        data=data,
        y=metrics[0],
        y_label=metrics[0],
        x=dimensions[0],
        x_label=dimensions[0],
        use_container_width=True,
        height=500,
    )
    CONFIG.query_info.render_duration = datetime.now() - render_start
    return scatterplot


col = st.columns((4, 2), gap="medium")

with col[0]:
    st.markdown("#### Visualization")
    if CONFIG.exception.text:
        st.error(CONFIG.exception.text)
    elif CONFIG.view_type == ChartType.US_MAP:
        choropleth = make_choropleth()
        st.plotly_chart(choropleth, use_container_width=True)
    elif CONFIG.view_type == ChartType.LINE_CHART:
        line_chart = make_line_chart()
    elif CONFIG.view_type == ChartType.BAR_CHART:
        bar_chart = make_bar_chart()
    elif CONFIG.view_type == ChartType.SCATTER_PLOT:
        scatter_plot = make_scatter_plot()
    else:
        table = make_table()

with col[1]:
    st.markdown(
        f"""#### Query Information
- SQL Size: {len(CONFIG.query_info.query_text)}
- Trilogy Size: {len(CONFIG.query_info.trilogy_text)}
- Parse Time: {CONFIG.query_info.parse_duration}
- Runtime: {CONFIG.query_info.duration}
- Render: {CONFIG.query_info.render_duration}
- Shape: {CONFIG.query_results.data.shape}

#### Query
```sql
{CONFIG.query_info.query_text}
``` 
#### Trilogy
```sql
{CONFIG.query_info.trilogy_text}
```
"""
    )
