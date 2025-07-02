import streamlit as st
import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional, Any

# LangChain imports with ChatGroq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL Application",
    page_icon="🤖",
    layout="wide"
)

class SQLOutputParser(BaseOutputParser):
    """Custom parser to extract SQL from LLM response"""
    
    def parse(self, text: str) -> str:
        # Look for SQL between ```sql and ``` or just clean the response
        if "```sql" in text:
            start = text.find("```sql") + 6
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        
        # If no code blocks, return the cleaned text
        lines = text.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH']):
                sql_lines.append(line)
            elif sql_lines and line:  # Continue collecting lines after SQL starts
                sql_lines.append(line)
        
        return ' '.join(sql_lines).strip() if sql_lines else text.strip()
    
    @property
    def _type(self) -> str:
        return "sql_output_parser"

class TextToSQLChain:
    """LangChain-based Text-to-SQL converter using ChatGroq v0.1.3"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = None
        self.chain = None
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain with ChatGroq v0.1.3"""
        try:
            # Initialize ChatGroq with correct parameters for version 0.1.3
            self.llm = ChatGroq(
                groq_api_key=self.api_key,  # Correct parameter for v0.1.3
                model_name="llama-3.3-70b-versatile",  # Correct parameter for v0.1.3
                temperature=0.1
                # Note: max_tokens not supported in this version
            )
            
            # Create prompt template with better context
            prompt_template = """You are an expert SQL Server query generator. Convert the natural language question into a SQL Server T-SQL query.

Database Schema and Sample Data:
{schema}

IMPORTANT GUIDELINES:
1. Use SQL Server T-SQL syntax (not SQLite, MySQL, or PostgreSQL)
2. Use square brackets around table and column names: [schema].[table_name]
3. Use TOP N instead of LIMIT for row limits
4. Always specify schema names when referencing tables
5. When user asks general questions like "show me data" or "what's in the database", query actual data tables (not metadata tables)
6. For exploration queries, use TOP 100 to limit results
7. Look for tables with actual business data, not just metadata tables
8. If searching for specific values, try LIKE '%value%' for partial matches
9. Generate queries that are likely to return actual data

Current database: {database}
Server: {server}

Examples of good queries for data exploration:
- SELECT TOP 100 * FROM [dbo].[largest_table_name]
- SELECT COUNT(*) as RecordCount, 'table_name' as TableName FROM [schema].[table_name]
- SELECT DISTINCT [column_name] FROM [schema].[table_name] WHERE [column_name] IS NOT NULL

Human Question: {question}

SQL Server T-SQL Query:"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["schema", "database", "server", "question"]
            )
            
            # Create the chain
            self.chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_parser=SQLOutputParser()
            )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error setting up LangChain with ChatGroq: {str(e)}")
            st.error("🔧 For langchain-groq==0.1.3, ensure you're using: groq_api_key and model_name parameters")
            return False
    
    def generate_sql(self, question: str, schema: str, database: str, server: str) -> str:
        """Generate SQL using LangChain with ChatGroq"""
        try:
            if not self.chain:
                return "Error: LangChain not properly initialized"
            
            result = self.chain.run(
                schema=schema,
                database=database,
                server=server,
                question=question
            )
            
            return result.strip()
            
        except Exception as e:
            return f"Error generating SQL: {str(e)}"

class DatabaseManager:
    """SQL Server database manager using Windows Authentication"""
    
    def __init__(self, server: str, database: str):
        self.server = server
        self.database = database
        self.connection = None
    
    def connect(self):
        """Connect to SQL Server using Windows Authentication"""
        try:
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
                f"Connection Timeout=30;"
                f"Command Timeout=30;"
            )
            
            self.connection = pyodbc.connect(connection_string, timeout=30)
            return True, "Connected successfully using Windows Authentication"
            
        except Exception as e:
            error_message = f"Windows Authentication failed: {str(e)}"
            return False, error_message

    def get_table_info_with_data(self):
        """Get enhanced table information including row counts and sample data"""
        if not self.connection:
            success, message = self.connect()
            if not success:
                return f"Connection failed: {message}"
        
        try:
            cursor = self.connection.cursor()
            
            # Get all tables with row counts
            cursor.execute("""
                SELECT 
                    s.name AS schema_name,
                    t.name AS table_name,
                    p.rows AS row_count
                FROM sys.tables t
                INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                INNER JOIN sys.partitions p ON t.object_id = p.object_id
                WHERE p.index_id IN (0,1)
                    AND s.name NOT IN ('sys', 'information_schema')
                ORDER BY p.rows DESC
            """)
            
            tables_with_counts = cursor.fetchall()
            return tables_with_counts
            
        except Exception as e:
            return f"Error getting table info: {str(e)}"

    def get_enhanced_schema(self):
        """Get enhanced database schema with row counts and sample data"""
        if not self.connection:
            success, message = self.connect()
            if not success:
                return f"Connection failed: {message}"
        
        try:
            cursor = self.connection.cursor()
            
            schema_info = [f"Database: {self.database} on Server: {self.server}\n"]
            
            # Get SQL Server version
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            schema_info.append(f"SQL Server Version: {version[:50]}...\n")
            
            # Get tables with row counts
            tables_with_counts = self.get_table_info_with_data()
            if isinstance(tables_with_counts, str):
                return tables_with_counts
            
            schema_info.append(f"Found {len(tables_with_counts)} tables with data:\n")
            
            # Show top 10 tables by row count with sample data
            for i, (schema_name, table_name, row_count) in enumerate(tables_with_counts[:10]):
                schema_info.append(f"\n{i+1}. Table: [{schema_name}].[{table_name}] ({row_count:,} rows)")
                
                # Get column info
                cursor.execute("""
                    SELECT TOP 5
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, schema_name, table_name)
                
                columns = cursor.fetchall()
                schema_info.append("   Columns:")
                for column in columns:
                    col_name, data_type, is_nullable = column
                    nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                    schema_info.append(f"     - [{col_name}]: {data_type} {nullable}")
                
                # Get sample data if table has data
                if row_count > 0:
                    try:
                        sample_query = f"SELECT TOP 3 * FROM [{schema_name}].[{table_name}]"
                        cursor.execute(sample_query)
                        sample_rows = cursor.fetchall()
                        if sample_rows:
                            schema_info.append(f"   Sample data (first 3 rows):")
                            for j, row in enumerate(sample_rows):
                                row_data = [str(val)[:50] + "..." if val and len(str(val)) > 50 else str(val) for val in row]
                                schema_info.append(f"     Row {j+1}: {row_data}")
                    except Exception:
                        schema_info.append("   (Sample data unavailable)")
            
            return '\n'.join(schema_info)
            
        except Exception as e:
            return f"Error getting enhanced schema: {str(e)}"

    def get_quick_data_overview(self):
        """Get a quick overview of data in the database"""
        if not self.connection:
            success, message = self.connect()
            if not success:
                return {"success": False, "error": f"Connection failed: {message}"}
        
        try:
            cursor = self.connection.cursor()
            
            # Get table counts
            cursor.execute("""
                SELECT 
                    s.name AS schema_name,
                    t.name AS table_name,
                    p.rows AS row_count
                FROM sys.tables t
                INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                INNER JOIN sys.partitions p ON t.object_id = p.object_id
                WHERE p.index_id IN (0,1)
                    AND s.name NOT IN ('sys', 'information_schema')
                    AND p.rows > 0
                ORDER BY p.rows DESC
            """)
            
            results = cursor.fetchall()
            
            if results:
                data = []
                for schema_name, table_name, row_count in results:
                    data.append({
                        'Schema': schema_name,
                        'Table': table_name,
                        'Row Count': f"{row_count:,}"
                    })
                
                return {
                    "success": True,
                    "data": data,
                    "total_tables": len(results)
                }
            else:
                return {"success": True, "data": [], "total_tables": 0}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def execute_query(self, sql: str):
        """Execute SQL query"""
        if not self.connection:
            success, message = self.connect()
            if not success:
                return {"success": False, "error": f"Connection failed: {message}"}
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            
            if sql.strip().upper().startswith('SELECT') or sql.strip().upper().startswith('WITH'):
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows)
                }
            else:
                affected_rows = cursor.rowcount
                self.connection.commit()
                return {
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"Query executed. {affected_rows} row(s) affected."
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Main Streamlit application"""
    
    st.title("🤖 Text-to-SQL Application")
    st.subheader("Convert natural language to SQL queries using LangChain & ChatGroq")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection method info
        st.subheader("🔐 Database Connection")
        st.success("✅ Using Windows Authentication")
        st.info("This method has been verified to work with your server")
        
        # API Key
        st.subheader("🔑 Groq API Key")
        api_key_source = st.radio(
            "API Key Source:",
            ["From .env file", "Manual input"]
        )
        
        if api_key_source == "From .env file":
            env_api_key = os.getenv('GROQ_API_KEY')
            if env_api_key and env_api_key != "your_groq_api_key_here":
                st.success("✅ API key loaded from .env file")
                api_key = env_api_key
                # Show masked key
                masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "***"
                st.code(f"Key: {masked_key}")
            else:
                st.error("❌ No API key found in .env file")
                api_key = None
        else:
            api_key = st.text_input(
                "Enter your Groq API Key:",
                type="password",
                help="Get your API key from https://console.groq.com/"
            )
            if api_key:
                st.success("✅ API key entered")
        
        # Database selection
        st.subheader("🗄️ Database Selection")
        selected_db = st.selectbox(
            "Select Database:",
            ["ReportingODS", "FSM_REPORT"],
            index=0
        )
        
        # Test connection
        if st.button("🔌 Test Connection"):
            with st.spinner("Testing Windows Authentication connection..."):
                db = DatabaseManager("UKSALD-MARS01", selected_db)
                success, message = db.connect()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # Main content
    if not api_key:
        st.warning("⚠️ Please configure your Groq API key in the sidebar to continue.")
        st.info("You can either:")
        st.write("1. Create a `.env` file with `GROQ_API_KEY=your_key_here`")
        st.write("2. Enter your API key manually in the sidebar")
        return
    
    # Initialize components
    with st.spinner("Initializing LangChain with ChatGroq v0.1.3..."):
        try:
            text_to_sql = TextToSQLChain(api_key)
            if text_to_sql.chain:
                st.success("✅ LangChain with ChatGroq v0.1.3 initialized successfully!")
            else:
                st.error("❌ Failed to initialize LangChain")
                return
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            return
    
    db_manager = DatabaseManager("UKSALD-MARS01", selected_db)
    
    # Data exploration section
    st.header("📊 Database Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Quick Data Overview"):
            with st.spinner("Getting data overview..."):
                overview = db_manager.get_quick_data_overview()
                if overview['success']:
                    if overview['data']:
                        st.subheader(f"📈 Tables with Data ({overview['total_tables']} total)")
                        df = pd.DataFrame(overview['data'])
                        st.dataframe(df, use_container_width=True)
                        
                        # Suggest some queries
                        st.subheader("💡 Suggested Questions:")
                        largest_table = overview['data'][0]['Table'] if overview['data'] else 'table_name'
                        st.write(f"• Show me data from {largest_table}")
                        st.write(f"• What are the columns in {largest_table}?")
                        st.write("• Show me the first 10 rows from the largest table")
                        st.write("• How many records are in each table?")
                    else:
                        st.warning("No tables with data found")
                else:
                    st.error(f"Error: {overview['error']}")
    
    with col2:
        if st.button("📋 Detailed Schema"):
            with st.spinner("Loading enhanced schema..."):
                schema = db_manager.get_enhanced_schema()
                with st.expander("📊 Database Schema with Sample Data", expanded=True):
                    st.text(schema)
    
    # Query input
    st.header("🔍 Ask Your Question")
    
    # Pre-filled example questions
    example_questions = [
        "Show me the first 10 rows from the largest table",
        "What tables have the most data?",
        "Show me a sample of data from any table",
        "How many records are in each table?",
        "What are the column names in the biggest table?"
    ]
    
    selected_example = st.selectbox("Choose an example question:", ["Custom question..."] + example_questions)
    
    if selected_example == "Custom question...":
        user_question = st.text_area(
            "Enter your question in natural language:",
            height=100,
            placeholder="e.g., Show me data from the largest table"
        )
    else:
        user_question = selected_example
        st.text_area(
            "Selected question:",
            value=selected_example,
            height=100,
            disabled=True
        )
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        execute_button = st.button("🚀 Execute Query", type="primary")
    
    with col2:
        generate_only = st.button("📝 Generate SQL Only")
    
    if execute_button or generate_only:
        if not user_question.strip():
            st.warning("⚠️ Please enter a question first.")
            return
        
        # Get enhanced schema for context
        with st.spinner("Getting database schema with sample data..."):
            schema = db_manager.get_enhanced_schema()
        
        # Generate SQL using LangChain with ChatGroq
        with st.spinner("🧠 Generating SQL query with enhanced context..."):
            sql_query = text_to_sql.generate_sql(
                question=user_question,
                schema=schema,
                database=selected_db,
                server="UKSALD-MARS01"
            )
        
        st.header("📄 Generated SQL Query")
        
        if sql_query.startswith("Error"):
            st.error(f"❌ {sql_query}")
            return
        
        st.code(sql_query, language="sql")
        
        if execute_button:
            # Execute query
            with st.spinner("⚡ Executing query..."):
                result = db_manager.execute_query(sql_query)
            
            st.header("📊 Query Results")
            
            if result['success']:
                if 'columns' in result and result['row_count'] > 0:
                    # Display results
                    df = pd.DataFrame(result['rows'], columns=result['columns'])
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📈 Rows Returned", result['row_count'])
                    with col2:
                        st.metric("📋 Columns", len(result['columns']))
                    with col3:
                        st.metric("🔗 Authentication", "Windows Auth")
                    
                    # Display data
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download as CSV",
                        csv,
                        "query_results.csv",
                        "text/csv"
                    )
                    
                elif 'columns' in result and result['row_count'] == 0:
                    st.info("✅ Query executed successfully, but no results found.")
                    st.warning("💡 Try these suggestions:")
                    st.write("• Use broader search terms (remove specific filters)")
                    st.write("• Try 'Show me data from the largest table'")
                    st.write("• Check the 'Quick Data Overview' to see available data")
                else:
                    st.success(f"✅ {result.get('message', 'Query executed successfully')}")
            else:
                st.error(f"❌ Query execution failed: {result['error']}")
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 Powered by LangChain & ChatGroq v0.1.3 | 🛠️ Built with Streamlit | 🗄️ SQL Server (Windows Auth)")

if __name__ == "__main__":
    main()