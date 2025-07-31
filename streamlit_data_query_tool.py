import streamlit as st
import os
import json
import logging
import numpy as np
import pandas as pd
import re
import math
import urllib.parse
import asyncio
import httpx
import duckdb
from pathlib import Path
from typing import List, Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io

# Configure Streamlit page
st.set_page_config(
    page_title="AI Data Query Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #44ff44;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Custom JSON encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

# System prompt for SQL generation
SYSTEM_PROMPT = (
    "You are an expert data analyst tasked with analyzing data using DuckDB SQL syntax. "
    "Based on the user's question, determine the appropriate analytical approach:\n\n"
    "CRITICAL: Column Name Handling\n"
    "1. ALWAYS use the EXACT column names as shown in the schema\n"
    "2. If a column name contains spaces or special characters, it MUST be quoted with double quotes\n"
    "3. Example: If schema shows \"Transaction Type\", use \"Transaction Type\" in queries\n"
    "4. Never modify or transform column names\n"
    "5. Always check the schema first to get the exact column names\n\n"
    "For questions about complaints/issues/problems:\n"
    "- Use a simple but effective approach with LIKE operators\n"
    "- Example query structure:\n"
    "  SELECT \"Review Text\", COUNT(*) as Frequency\n"
    "  FROM dataset\n"
    "  WHERE LOWER(\"Review Text\") LIKE '%keyword1%'\n"
    "     OR LOWER(\"Review Text\") LIKE '%keyword2%'\n"
    "  GROUP BY \"Review Text\"\n"
    "  ORDER BY Frequency DESC\n"
    "For trends or patterns:\n"
    "- Use simple aggregations (COUNT, AVG, SUM)\n"
    "- Group by relevant columns\n"
    "- Always use LOWER() for case-insensitive text matching\n"
    "- Always quote column names with spaces\n\n"
    "Important rules:\n"
    "1. Always use simple, DuckDB-compatible SQL syntax\n"
    "2. Avoid complex string manipulations\n"
    "3. Use straightforward GROUP BY and aggregations\n"
    "4. Never use LIMIT unless specifically requested\n"
    "5. Focus on finding meaningful patterns in the data\n"
    "6. When working with dates, use STRPTIME for conversions\n"
    "7. Use proper file path handling with forward slashes\n\n"
    "Always ensure queries are compatible with DuckDB and provide clear insights."
)

class StreamlitDataQueryTool:
    def __init__(self):
        """Initialize the Streamlit Data Query Tool."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.datasets = {}
            st.session_state.query_history = []
            st.session_state.ollama_connected = False
            st.session_state.current_dataset = None
            
        # Initialize DuckDB connection
        if 'duckdb_connection' not in st.session_state:
            st.session_state.duckdb_connection = duckdb.connect(":memory:")
            self._initialize_duckdb()
        
        self.con = st.session_state.duckdb_connection
        self.ollama_base_url = "http://localhost:11434"
        self.model = "llama3.1:8b"

    def _initialize_duckdb(self):
        """Initialize DuckDB and load required extensions."""
        try:
            extensions = ["excel", "mysql", "httpfs", "sqlite"]
            for ext in extensions:
                try:
                    self.con.execute(f"INSTALL {ext}")
                    self.con.execute(f"LOAD {ext}")
                except Exception as e:
                    st.sidebar.warning(f"Could not load extension {ext}: {e}")
        except Exception as e:
            st.sidebar.error(f"Error initializing DuckDB extensions: {e}")

    def normalize_file_path(self, file_path: str) -> str:
        """Normalize file path to use forward slashes."""
        if not file_path:
            return file_path
        
        normalized_path = file_path.replace('\\', '/')
        
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                normalized_path = str(path_obj.resolve()).replace('\\', '/')
        except:
            pass
            
        return normalized_path

    def quote_column_name(self, column_name: str) -> str:
        """Quote column names that contain spaces or special characters."""
        if not column_name:
            return column_name

        column_name = column_name.strip('"')

        if ' ' in column_name or any(c in column_name for c in '[](){}<>+-*/=!@#$%^&|\\'):
            return f'"{column_name}"'
        return column_name

    async def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model['name'] for model in models]
                    
                    if self.model not in model_names:
                        st.error(f"Model '{self.model}' not found! Please run: `ollama pull {self.model}`")
                        return False
                    
                    st.session_state.ollama_connected = True
                    return True
                else:
                    st.error(f"Ollama responded with status: {response.status_code}")
                    return False
        except Exception as e:
            st.error(f"Cannot connect to Ollama: {e}")
            st.error("Please make sure Ollama is running and the model is installed.")
            return False

    async def call_ollama_api(self, user_input: str, custom_system_prompt: Optional[str] = None) -> str:
        """Call Ollama API with the given user input."""
        current_prompt = custom_system_prompt if custom_system_prompt else SYSTEM_PROMPT
        full_prompt = f"{current_prompt}\n\nUser Query: {user_input}\n\nPlease provide a response:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 2048,
                "stop": ["User:", "System:"]
            }
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.ollama_base_url}/api/generate", json=payload)
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API returned status {response.status_code}: {response.text}")

                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            raise Exception(f"Error calling Ollama API: {str(e)}")

    def is_remote_url(self, file_path: str) -> bool:
        """Check if the given path is a remote URL."""
        try:
            result = urllib.parse.urlparse(file_path)
            return all([result.scheme, result.netloc])
        except:
            return False

    def get_schema_from_duckdb(self, file_path: str) -> tuple[str, list]:
        """Get schema using DuckDB's introspection capabilities."""
        normalized_path = self.normalize_file_path(file_path)
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension in [".csv", ".txt"]:
                schema_info = self.con.execute(
                    f"DESCRIBE SELECT * FROM read_csv_auto('{normalized_path}') LIMIT 0").fetchall()
                sample_data = self.con.execute(f"SELECT * FROM read_csv_auto('{normalized_path}') LIMIT 5").fetchall()
            elif file_extension == ".parquet":
                schema_info = self.con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{normalized_path}') LIMIT 0").fetchall()
                sample_data = self.con.execute(f"SELECT * FROM read_parquet('{normalized_path}') LIMIT 5").fetchall()
            elif file_extension == ".json":
                schema_info = self.con.execute(
                    f"DESCRIBE SELECT * FROM read_json_auto('{normalized_path}') LIMIT 0").fetchall()
                sample_data = self.con.execute(f"SELECT * FROM read_json_auto('{normalized_path}') LIMIT 5").fetchall()
            elif file_extension == ".xlsx":
                schema_info = self.con.execute(f"DESCRIBE SELECT * FROM read_excel('{normalized_path}') LIMIT 0").fetchall()
                sample_data = self.con.execute(f"SELECT * FROM read_excel('{normalized_path}') LIMIT 5").fetchall()
            elif file_extension == ".db":
                tables = self.con.execute(
                    f"SELECT name FROM sqlite_scan('{normalized_path}', 'sqlite_master') WHERE type='table'").fetchall()
                if not tables:
                    raise ValueError("No tables found in SQLite database")
                table_name = tables[0][0]
                read_function = f"sqlite_scan('{normalized_path}', '{table_name}')"
                schema_info = self.con.execute(f"DESCRIBE SELECT * FROM {read_function} LIMIT 0").fetchall()
                sample_data = self.con.execute(f"SELECT * FROM {read_function} LIMIT 5").fetchall()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            schema_description = (
                "CREATE TABLE dataset (\n"
                + ",\n".join([f"[{col[0]}] {col[1]}" for col in schema_info])
                + "\n);"
            )

            return schema_description, sample_data

        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

    def process_sql_query(self, sql_query: str, schema_info: list) -> str:
        """Process SQL query to properly quote column names and fix table references."""
        if not st.session_state.datasets:
            return sql_query

        # Get the actual read function
        current_dataset = st.session_state.current_dataset
        if current_dataset and current_dataset in st.session_state.datasets:
            read_function = st.session_state.datasets[current_dataset]['read_function']
            
            # Replace 'dataset' with actual read function
            sql_query = re.sub(r'\bFROM\s+dataset\b', f'FROM {read_function}', sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r'\bJOIN\s+dataset\b', f'JOIN {read_function}', sql_query, flags=re.IGNORECASE)

        # Process column names
        if schema_info:
            column_mapping = {}
            for col in schema_info:
                if col and len(col) > 0 and col[0]:
                    original = col[0].strip('"')
                    quoted = self.quote_column_name(original)
                    if original and quoted:
                        column_mapping[original] = quoted

            sorted_columns = sorted(column_mapping.keys(), key=len, reverse=True)
            for original in sorted_columns:
                quoted = column_mapping[original]
                pattern = r'(?<!\w)' + re.escape(original) + r'(?!\w)'
                sql_query = re.sub(pattern, quoted, sql_query)

        return sql_query

    async def load_dataset(self, file_path: str = None, uploaded_file = None) -> bool:
        """Load a dataset and store its information."""
        try:
            if uploaded_file is not None:
                # Handle uploaded file
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                file_path = temp_path
                dataset_name = os.path.splitext(uploaded_file.name)[0]
            else:
                # Handle file path
                if not file_path or not os.path.exists(file_path):
                    st.error(f"File not found: {file_path}")
                    return False
                
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]

            # Normalize dataset name
            dataset_name = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_name)
            if dataset_name[0].isdigit():
                dataset_name = f"t_{dataset_name}"

            # Get schema and sample data
            schema_description, sample_data = self.get_schema_from_duckdb(file_path)
            
            # Determine read function
            normalized_path = self.normalize_file_path(file_path)
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension in ['.csv', '.txt']:
                read_function = f"read_csv_auto('{normalized_path}')"
            elif file_extension == '.parquet':
                read_function = f"read_parquet('{normalized_path}')"
            elif file_extension == '.json':
                read_function = f"read_json_auto('{normalized_path}')"
            elif file_extension == '.xlsx':
                read_function = f"read_excel('{normalized_path}')"
            elif file_extension == '.db':
                tables = self.con.execute(
                    f"SELECT name FROM sqlite_scan('{normalized_path}', 'sqlite_master') WHERE type='table'").fetchall()
                if not tables:
                    raise ValueError("No tables found in SQLite database")
                table_name = tables[0][0]
                read_function = f"sqlite_scan('{normalized_path}', '{table_name}')"

            # Store dataset info
            st.session_state.datasets[dataset_name] = {
                "schema_description": schema_description,
                "file_path": normalized_path,
                "read_function": read_function,
                "sample_data": sample_data
            }
            
            st.session_state.current_dataset = dataset_name
            
            return True

        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False

    async def execute_query(self, user_query: str) -> Dict[str, Any]:
        """Execute a natural language query on the loaded datasets."""
        try:
            if not st.session_state.datasets or not st.session_state.current_dataset:
                return {"error": "No datasets loaded. Please load a dataset first."}

            current_dataset = st.session_state.current_dataset
            dataset_info = st.session_state.datasets[current_dataset]
            
            # Prepare dataset schema for LLM
            schema_description = dataset_info.get("schema_description")
            read_function = dataset_info.get("read_function")
            
            llm_prompt = (
                f"Dataset schema: {schema_description}\n"
                f"CRITICAL: Use '{read_function}' as the table name in your SQL query\n"
                f"Please write a DuckDB query for: {user_query}"
            )

            # Call Ollama with the prompt
            llm_response = await self.call_ollama_api(llm_prompt)

            # Extract SQL query from response
            sql_query_match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL)
            if not sql_query_match:
                sql_query_match = re.search(r"```\n(.*?)\n```", llm_response, re.DOTALL)
                if not sql_query_match:
                    return {"error": "Failed to extract SQL query from the LLM response.", "llm_response": llm_response}

            sql_query = sql_query_match.group(1).strip()

            # Get schema info for column name processing
            schema_info = self.con.execute(f"DESCRIBE SELECT * FROM {read_function} LIMIT 0").fetchall()
            
            # Process the SQL query
            sql_query = self.process_sql_query(sql_query, schema_info)

            # Execute the query
            result_df = self.con.execute(sql_query).fetchdf()

            # Process results
            result_df = result_df.apply(
                lambda col: col.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            )

            # Add to query history
            st.session_state.query_history.append({
                "timestamp": datetime.now(),
                "query": user_query,
                "sql": sql_query,
                "rows": len(result_df)
            })

            return {
                "success": True,
                "sql_query": sql_query,
                "result_df": result_df,
                "llm_response": llm_response
            }

        except Exception as e:
            return {"error": f"Error executing query: {str(e)}"}

    def create_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        """Create appropriate visualizations based on the data."""
        visualizations = []
        
        if df.empty:
            return visualizations

        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Limit to reasonable number of rows for visualization
        if len(df) > 1000:
            df_viz = df.sample(1000)
        else:
            df_viz = df

        # Bar chart for categorical data
        if categorical_cols and len(categorical_cols) >= 1:
            col = categorical_cols[0]
            if df_viz[col].nunique() <= 20:  # Reasonable number of categories
                value_counts = df_viz[col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"Distribution of {col}",
                    labels={'x': col, 'y': 'Count'}
                )
                visualizations.append({
                    "type": "bar",
                    "title": f"Distribution of {col}",
                    "figure": fig
                })

        # Histogram for numeric data
        if numeric_cols:
            for col in numeric_cols[:2]:  # Limit to 2 histograms
                fig = px.histogram(
                    df_viz, 
                    x=col,
                    title=f"Distribution of {col}",
                    nbins=30
                )
                visualizations.append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "figure": fig
                })

        # Scatter plot for two numeric columns
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                df_viz,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
            )
            visualizations.append({
                "type": "scatter",
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
                "figure": fig
            })

        return visualizations

def main():
    """Main function to run the Streamlit app."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1> AI-Powered Data Query Tool</h1>
        <p>Upload your data and ask questions in natural language!</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize the tool
    tool = StreamlitDataQueryTool()

    # Sidebar for configuration and dataset info
    with st.sidebar:
        st.header(" Configuration")
        
        # Ollama connection status
        if st.button("Check Ollama Connection"):
            with st.spinner("Checking Ollama connection..."):
                connected = asyncio.run(tool.check_ollama_connection())
                if connected:
                    st.success(" Ollama is connected!")
                else:
                    st.error(" Ollama connection failed!")

        st.markdown("---")
        
        # Dataset upload/selection
        st.header("Dataset Management")
        
        upload_method = st.radio("Choose upload method:", ["Upload File", "File Path"])
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a data file",
                type=['csv', 'xlsx', 'json', 'parquet', 'db'],
                help="Supported formats: CSV, Excel, JSON, Parquet, SQLite"
            )
            
            if uploaded_file is not None:
                if st.button("Load Dataset"):
                    with st.spinner("Loading dataset..."):
                        success = asyncio.run(tool.load_dataset(uploaded_file=uploaded_file))
                        if success:
                            st.success(" Dataset loaded successfully!")
                        else:
                            st.error(" Failed to load dataset!")
        
        else:
            file_path = st.text_input("Enter file path:", help="Enter the full path to your data file")
            if file_path and st.button("Load Dataset"):
                with st.spinner("Loading dataset..."):
                    success = asyncio.run(tool.load_dataset(file_path=file_path))
                    if success:
                        st.success("Dataset loaded successfully!")
                    else:
                        st.error(" Failed to load dataset!")

        # Dataset info
        if st.session_state.datasets and st.session_state.current_dataset:
            st.markdown("---")
            st.header("Current Dataset")
            current_dataset = st.session_state.current_dataset
            dataset_info = st.session_state.datasets[current_dataset]
            
            st.write(f"**Dataset:** {current_dataset}")
            
            # Show schema
            with st.expander("View Schema"):
                st.code(dataset_info["schema_description"], language="sql")
            
            # Show sample data
            if "sample_data" in dataset_info and dataset_info["sample_data"]:
                with st.expander("Sample Data (First 5 rows)"):
                    sample_df = pd.DataFrame(dataset_info["sample_data"])
                    st.dataframe(sample_df)

        # Query history
        if st.session_state.query_history:
            st.markdown("---")
            st.header(" Query History")
            for i, query in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
                with st.expander(f"Query {len(st.session_state.query_history)-i}: {query['query'][:30]}..."):
                    st.write(f"**Time:** {query['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**Query:** {query['query']}")
                    st.write(f"**Rows returned:** {query['rows']}")
                    st.code(query['sql'], language="sql")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(" Ask Your Question")
        
        # Sample questions based on loaded dataset
        if st.session_state.datasets and st.session_state.current_dataset:
            st.write("**Example questions you can ask:**")
            sample_questions = [
                "How many rows are in this dataset?",
                "What are the column names and their data types?",
                "Show me the first 10 rows",
                "What are the unique values in each categorical column?",
                "Are there any missing values?",
                "What's the summary statistics for numeric columns?"
            ]
            
            for question in sample_questions:
                if st.button(f" {question}", key=f"sample_{question}"):
                    st.session_state.user_query = question

        # Query input
        user_query = st.text_area(
            "Enter your question about the data:",
            height=100,
            placeholder="e.g., 'Show me the top 10 customers by sales amount' or 'What's the average price by category?'",
            key="query_input"
        )

        # Execute query button
        if st.button("Execute Query", type="primary"):
            if not user_query:
                st.warning("Please enter a question!")
            elif not st.session_state.datasets:
                st.warning("Please load a dataset first!")
            elif not st.session_state.ollama_connected:
                st.warning("Please check Ollama connection first!")
            else:
                with st.spinner("Processing your query..."):
                    result = asyncio.run(tool.execute_query(user_query))
                    
                    if "error" in result:
                        st.markdown(f"""
                        <div class="error-box">
                            <h4> Error</h4>
                            <p>{result['error']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "llm_response" in result:
                            with st.expander("View LLM Response"):
                                st.text(result["llm_response"])
                    
                    else:
                        # Display success
                        st.markdown(f"""
                        <div class="success-box">
                            <h4> Query Executed Successfully</h4>
                            <p>Found {len(result['result_df'])} rows</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show generated SQL
                        with st.expander("🔍 View Generated SQL"):
                            st.code(result['sql_query'], language="sql")
                        
                        # Display results
                        st.header("📊 Results")
                        
                        df = result['result_df']
                        
                        if df.empty:
                            st.info("No results found for your query.")
                        else:
                            # Summary metrics
                            col1_metric, col2_metric, col3_metric = st.columns(3)
                            with col1_metric:
                                st.metric("Total Rows", len(df))
                            with col2_metric:
                                st.metric("Total Columns", len(df.columns))
                            with col3_metric:
                                numeric_cols = df.select_dtypes(include=[np.number]).columns
                                st.metric("Numeric Columns", len(numeric_cols))
                            
                            # Data table
                            st.subheader(" Data Table")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label=" Download CSV",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Visualizations
                            st.subheader(" Visualizations")
                            visualizations = tool.create_visualizations(df)
                            
                            if visualizations:
                                for viz in visualizations:
                                    st.plotly_chart(viz["figure"], use_container_width=True)
                            else:
                                st.info("No suitable visualizations available for this data.")
                            
                            # Summary statistics for numeric columns
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                st.subheader(" Summary Statistics")
                                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    with col2:

        
        # System status
        st.markdown("---")
        st.header(" System Status")
        
        # Connection status
        if st.session_state.get('ollama_connected', False):
            st.success("🟢 Ollama Connected")
        else:
            st.error("🔴 Ollama Disconnected")
        
        # Dataset status
        if st.session_state.datasets:
            st.success(f"🟢 Dataset Loaded ({len(st.session_state.datasets)} total)")
        else:
            st.warning("🟡 No Dataset Loaded")
        
        # Query count
        query_count = len(st.session_state.query_history)
        st.info(f" Queries Executed: {query_count}")

if __name__ == "__main__":
    main()