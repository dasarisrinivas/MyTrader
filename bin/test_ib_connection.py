#!/usr/bin/env python3
"""Simple IB Gateway connection test script."""

import asyncio
from ib_insync import IB, util

async def test_connection():
    """Test basic IB Gateway connection."""
    ib = IB()
    
    print("=" * 60)
    print("IB Gateway Connection Test")
    print("=" * 60)
    print()
    
    # Try different client IDs
    for client_id in [3, 4, 5, 10, 100]:
        print(f"Attempting connection with client_id={client_id}...")
        try:
            await asyncio.wait_for(
                ib.connectAsync('127.0.0.1', 4002, clientId=client_id, timeout=5),
                timeout=10
            )
            print(f"✅ SUCCESS! Connected with client_id={client_id}")
            print(f"   Server version: {ib.client.serverVersion()}")
            print(f"   Connection time: {ib.client.connTime}")
            
            # Test getting account info
            accounts = ib.managedAccounts()
            print(f"   Accounts: {accounts}")
            
            # Try to get a contract
            contract = await ib.qualifyContractsAsync(
                util.Stock('AAPL', 'SMART', 'USD')
            )
            if contract:
                print(f"   ✓ Contract qualified: {contract[0].symbol}")
            
            print()
            print(f"🎉 Use client_id={client_id} in your main.py")
            ib.disconnect()
            return client_id
            
        except asyncio.TimeoutError:
            print(f"   ✗ Timeout with client_id={client_id}")
        except ConnectionRefusedError:
            print(f"   ✗ Connection refused - IB Gateway may not be running")
            return None
        except Exception as e:
            print(f"   ✗ Error with client_id={client_id}: {e}")
        
        if ib.isConnected():
            ib.disconnect()
        
        await asyncio.sleep(1)
    
    print()
    print("❌ All client IDs failed!")
    print()
    print("Troubleshooting steps:")
    print("1. Check IB Gateway is running")
    print("2. Check API settings: Configure > Settings > API > Settings")
    print("   - Enable ActiveX and Socket Clients: ✓")
    print("   - Socket port: 4002")
    print("   - Master API client ID: (leave blank or set to 0)")
    print("   - Read-Only API: ✗ (unchecked)")
    print("   - Trusted IPs: 127.0.0.1")
    print("3. Restart IB Gateway")
    print("4. Check firewall settings")
    return None

if __name__ == "__main__":
    asyncio.run(test_connection())
