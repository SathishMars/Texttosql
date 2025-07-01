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
            
            # Create prompt template
            prompt_template = """You are an expert SQL Server query generator. Convert the natural language question into a SQL Server T-SQL query.

Database Schema:
{schema}

SQL Server T-SQL Rules:
1. Use SQL Server T-SQL syntax (not SQLite, MySQL, or PostgreSQL)
2. Use square brackets around table and column names: [schema].[table_name]
3. Use TOP N instead of LIMIT for row limits
4. Use appropriate SQL Server data types and functions
5. For date operations, use SQL Server date functions like GETDATE(), DATEADD(), DATEDIFF()
6. Always specify schema names when referencing tables
7. Use proper T-SQL syntax for JOINs, CTEs, and window functions
8. Generate clean, efficient T-SQL queries
9. Return only the SQL query, nothing else

Current database: {database}
Server: {server}

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
    """SQL Server database manager"""
    
    def __init__(self, server: str, database: str):
        self.server = server
        self.database = database
        self.connection = None
    
    def connect(self):
        """Connect to SQL Server"""
        try:
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )
            self.connection = pyodbc.connect(connection_string)
            return True, "Connected successfully"
        except Exception as e:
            return False, str(e)
    
    def get_schema(self):
        """Get database schema information"""
        if not self.connection:
            success, message = self.connect()
            if not success:
                return f"Connection failed: {message}"
        
        try:
            cursor = self.connection.cursor()
            
            schema_info = [f"Database: {self.database} on Server: {self.server}\n"]
            
            # Get tables and columns
            cursor.execute("""
                SELECT 
                    TABLE_SCHEMA,
                    TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
                    AND TABLE_SCHEMA NOT IN ('sys', 'information_schema')
                ORDER BY TABLE_SCHEMA, TABLE_NAME
            """)
            
            tables = cursor.fetchall()
            
            for table in tables[:]:  # Limit to first 15 tables
                schema_name, table_name = table
                schema_info.append(f"\nTable: {schema_name}.{table_name}")
                
                # Get columns for this table
                cursor.execute("""
                    SELECT TOP 10
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, schema_name, table_name)
                
                columns = cursor.fetchall()
                for column in columns:
                    col_name, data_type, is_nullable = column
                    nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                    schema_info.append(f"    - {col_name}: {data_type} {nullable}")
            
            return '\n'.join(schema_info)
            
        except Exception as e:
            return f"Error getting schema: {str(e)}"
    
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
        
        # Version info
        st.subheader("📦 Package Versions")
        try:
            import langchain
            import langchain_groq
            st.code(f"langchain: {langchain.__version__}")
            st.code(f"langchain-groq: {langchain_groq.__version__}")
            
            # Show parameter info for this version
            if hasattr(langchain_groq, '__version__') and langchain_groq.__version__.startswith('0.1.3'):
                st.success("✅ Optimized for langchain-groq v0.1.3")
                st.info("Using: groq_api_key + model_name parameters")
        except:
            st.warning("Cannot display version info")
        
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
        
        # Connection info
        st.subheader("🔗 Connection Info")
        st.info(f"**Server:** UKSALD-MARS01\n**Database:** {selected_db}\n**Auth:** Windows Authentication")
        
        # Model info
        st.subheader("🧠 AI Model")
        st.info("**Model:** Llama 3.1 70B Versatile\n**Provider:** Groq\n**Framework:** LangChain\n**Client:** ChatGroq v0.1.3")
        
        # Test connection
        if st.button("🔌 Test Connection"):
            if api_key:
                with st.spinner("Testing connection..."):
                    db = DatabaseManager("UKSALD-MARS01", selected_db)
                    success, message = db.connect()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Please provide API key first")
    
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
                st.info("Using groq_api_key + model_name parameters for v0.1.3")
            else:
                st.error("❌ Failed to initialize LangChain")
                return
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            return
    
    db_manager = DatabaseManager("UKSALD-MARS01", selected_db)
    
    # Database schema viewer
    with st.expander("📊 View Database Schema", expanded=False):
        if st.button("🔄 Load Schema"):
            with st.spinner("Loading schema..."):
                schema = db_manager.get_schema()
                st.text(schema)
    
    # Query input
    st.subheader("🔍 Ask Your Question")
    user_question = st.text_area(
        "Enter your question in natural language:",
        height=100,
        placeholder="e.g., Show me all tables in the current database"
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
        
        # Get schema for context
        with st.spinner("Getting database schema..."):
            schema = db_manager.get_schema()
        
        # Generate SQL using LangChain with ChatGroq
        with st.spinner("🧠 Generating SQL query with ChatGroq v0.1.3..."):
            sql_query = text_to_sql.generate_sql(
                question=user_question,
                schema=schema,
                database=selected_db,
                server="UKSALD-MARS01"
            )
        
        st.subheader("📄 Generated SQL Query")
        
        if sql_query.startswith("Error"):
            st.error(f"❌ {sql_query}")
            return
        
        st.code(sql_query, language="sql")
        
        if execute_button:
            # Execute query
            with st.spinner("⚡ Executing query..."):
                result = db_manager.execute_query(sql_query)
            
            st.subheader("📊 Query Results")
            
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
                        st.metric("🔗 Framework", "ChatGroq v0.1.3")
                    
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
                else:
                    st.success(f"✅ {result.get('message', 'Query executed successfully')}")
            else:
                st.error(f"❌ Query execution failed: {result['error']}")
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 Powered by LangChain & ChatGroq v0.1.3 | 🛠️ Built with Streamlit | 🗄️ SQL Server Integration")

if __name__ == "__main__":
    main()