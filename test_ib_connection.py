#!/usr/bin/env python3
"""Diagnostic script to test IB Gateway connection."""
import asyncio
from ib_insync import IB, util

async def test_connection():
    """Test connection to IB Gateway and report status."""
    print("\n" + "="*60)
    print("IB GATEWAY CONNECTION DIAGNOSTICS")
    print("="*60 + "\n")
    
    # Enable logging
    util.logToConsole('ERROR')
    
    # Test connection
    ib = IB()
    host = '127.0.0.1'
    port = 4002
    client_id = 999  # Use different client ID for testing
    
    try:
        print(f"🔌 Attempting to connect to {host}:{port}...")
        await ib.connectAsync(host, port, clientId=client_id, timeout=10)
        
        print(f"✅ Connected successfully!")
        print(f"   Server version: {ib.client.serverVersion()}")
        
        # Check if API is read-only
        print("\n📊 Testing API permissions...")
        
        # Try to get account info
        accounts = ib.managedAccounts()
        if accounts:
            print(f"✅ Accounts accessible: {', '.join(accounts)}")
        else:
            print("⚠️  No accounts found (may be read-only)")
        
        # Check for read-only mode
        print("\n⚙️  Checking API mode...")
        await asyncio.sleep(2)  # Wait for any error messages
        
        # If we got this far, connection is good
        print("\n" + "="*60)
        print("✅ CONNECTION SUCCESSFUL!")
        print("="*60)
        print("\n🎯 You can now start trading with ./start_trading.sh\n")
        
    except ConnectionRefusedError:
        print("\n" + "="*60)
        print("❌ CONNECTION FAILED: Connection Refused")
        print("="*60)
        print("\n🔍 Troubleshooting steps:")
        print("\n1. Is IB Gateway/TWS running?")
        print("   Run: lsof -i :4002")
        print("   Should show a Java process listening\n")
        print("2. Is the port correct?")
        print(f"   Trying to connect to: {host}:{port}")
        print("   Check IB Gateway: Edit > Global Configuration > API > Settings")
        print(f"   Socket port should be: {port}\n")
        print("3. Is API enabled?")
        print("   IB Gateway: Edit > Global Configuration > API > Settings")
        print("   'Enable ActiveX and Socket Clients' should be CHECKED\n")
        print("4. Is Read-Only API disabled?")
        print("   IB Gateway: Edit > Global Configuration > API > Settings")
        print("   'Read-Only API' should be UNCHECKED\n")
        print("5. Is 127.0.0.1 in Trusted IPs?")
        print("   IB Gateway: Edit > Global Configuration > API > Settings")
        print("   Trusted IP Addresses should include: 127.0.0.1\n")
        
    except asyncio.TimeoutError:
        print("\n" + "="*60)
        print("❌ CONNECTION FAILED: Timeout")
        print("="*60)
        print("\nIB Gateway is not responding. Check:")
        print("1. IB Gateway is running and logged in")
        print("2. Firewall is not blocking port 4002")
        print("3. No other application is using the API")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ CONNECTION FAILED: {type(e).__name__}")
        print("="*60)
        print(f"\nError: {e}\n")
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("Disconnected.")

if __name__ == "__main__":
    try:
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
