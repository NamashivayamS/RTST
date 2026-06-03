import sys
import traceback

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    print("Attempting to import RouterService...")
    from services.router_service import RouterService
    print("Import successful!")
except Exception as e:
    print(f"IMPORT ERROR CAUGHT: {e}")
    traceback.print_exc(file=sys.stdout)
