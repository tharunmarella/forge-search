# Testing the Codebase Search Tool

This directory contains test scripts to verify the `codebase_search` tool is working correctly.

## Prerequisites

1. **PostgreSQL with pgvector**: The database must be running and have the pgvector extension installed.
   ```bash
   # Install pgvector in your database
   psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

2. **Environment Variables**: Create a `.env` file with the following:
   ```bash
   # Required
   DATABASE_URL=postgresql://user:password@localhost:5432/forge_search
   VOYAGE_API_KEY=your-voyage-api-key-here
   
   # Optional
   VOYAGE_MODEL=voyage-code-3
   ```

3. **Dependencies**: Install all Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Test Scripts

### 1. Simple Test (Recommended for Quick Checks)

**File**: `test_codebase_search_simple.py`

This script runs a series of focused tests in sequence:
- Module imports
- Environment configuration
- Database connection
- Embedding generation
- Vector search
- Full codebase_search function

```bash
python test_codebase_search_simple.py
```

**Output**: Clear pass/fail for each test component with helpful error messages.

### 2. Comprehensive Test

**File**: `test_codebase_search.py`

This script runs a full test suite with multiple search queries and detailed output:
- Validates all dependencies
- Tests various search queries
- Tests component-focused searches
- Shows result previews

```bash
python test_codebase_search.py
```

**Output**: Detailed results including snippets of found code.

## Common Issues and Solutions

### Issue: "VOYAGE_API_KEY not set"

**Solution**: Get an API key from [Voyage AI](https://dash.voyageai.com/) and add it to your `.env` file:
```bash
VOYAGE_API_KEY=pa-xxxxx...
```

### Issue: "Database connection failed"

**Solution**: 
1. Check PostgreSQL is running: `pg_isready`
2. Verify DATABASE_URL in `.env`
3. Test connection manually:
   ```bash
   psql "postgresql://user:password@localhost:5432/forge_search"
   ```

### Issue: "No indexed data found"

**Solution**: You need to index your codebase first:
1. Start the server: `python -m app.main`
2. Use the indexing API endpoint
3. Or run the indexing helper scripts in `app/api/indexing_helpers.py`

### Issue: "ImportError" or "ModuleNotFoundError"

**Solution**: 
1. Make sure you're in the project root directory
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Issue: "pgvector extension not found"

**Solution**:
1. Install pgvector: https://github.com/pgvector/pgvector
2. Create the extension in your database:
   ```bash
   psql -d forge_search -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

## Interpreting Results

### ✅ All Tests Pass
Your codebase_search tool is working correctly! You can use it in your application.

### ❌ Some Tests Fail
Check the specific error messages. Common issues:
- **Environment**: Missing API keys or DATABASE_URL
- **Database**: Connection issues or missing pgvector extension
- **Indexing**: No data in the database (needs indexing)
- **API**: Voyage API issues (quota, invalid key, network)

### ⚠️ No Results Found
This usually means:
1. The workspace hasn't been indexed yet
2. The workspace ID is incorrect (default: "forge-search")
3. The query doesn't match any code in the database

## Updating Workspace ID

If your workspace has a different ID, edit the test files:

```python
workspace_id = "your-workspace-id"  # Change this line
```

## Next Steps

After tests pass:
1. Try searching with your own queries
2. Index your entire codebase
3. Test the search through the API
4. Integrate with your IDE or tools

## Troubleshooting

If you're still having issues after trying the solutions above:

1. **Enable Debug Logging**: Set log level to DEBUG in the test scripts:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Database Schema**: Ensure tables exist:
   ```sql
   \dt  -- in psql
   SELECT COUNT(*) FROM symbols;  -- Should return a number
   ```

3. **Test Voyage API Directly**: Try the embedding module in isolation:
   ```python
   from app.core import embeddings
   result = await embeddings.embed_query("test")
   print(len(result))  # Should print 1024
   ```

4. **Check Server Logs**: If running through the API, check server logs for errors.

## Performance Notes

- **First Run**: May be slower due to model loading and connection setup
- **Subsequent Runs**: Should be faster with warm connections
- **Large Workspaces**: Search time scales with database size (use HNSW indexing for speed)

## Additional Resources

- [Voyage AI Documentation](https://docs.voyageai.com/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [PostgreSQL Async with Python](https://www.psycopg.org/psycopg3/docs/)
