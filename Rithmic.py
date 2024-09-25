import asyncio
import websockets # type: ignore
import sys

from pb2 import base_pb2

from pb2 import request_heartbeat_pb2, response_heartbeat_pb2

from pb2 import request_login_pb2, response_login_pb2
from pb2 import request_logout_pb2, response_logout_pb2

from pb2 import request_subscribe_for_order_updates_pb2, response_subscribe_for_order_updates_pb2
from pb2 import exchange_order_notification_pb2

from pb2 import request_market_data_update_pb2, response_market_data_update_pb2

from pb2 import best_bid_offer_pb2, last_trade_pb2

from pb2 import request_rithmic_system_info_pb2, response_rithmic_system_info_pb2


async def connect(uri, ssl_context):
    ws = await websockets.connect(uri, ssl=ssl_context, ping_interval=3)
    print(f"connected to {uri}")

    return (ws)

async def subscribe_market(ws, exchange, symbol):
    rq = request_market_data_update_pb2.RequestMarketDataUpdate()

    rq.template_id      = 100;
    rq.user_msg.append("H7H Market Subcribed")

    rq.symbol      = symbol
    rq.exchange    = exchange
    rq.request     = request_market_data_update_pb2.RequestMarketDataUpdate.Request.SUBSCRIBE
    rq.update_bits = request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.LAST_TRADE | request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.BBO

    serialized = rq.SerializeToString()

    buf  = bytearray()
    buf  = serialized

    await ws.send(buf)

async def unsubscribe_market(ws, exchange, symbol):

    rq = request_market_data_update_pb2.RequestMarketDataUpdate()

    rq.template_id      = 100;
    rq.user_msg.append("H7H Market Unsubscribe")

    rq.symbol      = symbol
    rq.exchange    = exchange
    rq.request     = request_market_data_update_pb2.RequestMarketDataUpdate.Request.UNSUBSCRIBE
    rq.update_bits = request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.LAST_TRADE | request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.BBO

    serialized = rq.SerializeToString()

    buf  = bytearray()
    buf  = serialized

    await ws.send(buf)

async def subscribe_orders(ws, exchange, symbol):
    rq = request_subscribe_for_order_updates_pb2.RequestSubscribeForOrderUpdates()

    rq.template_id = 308;
    rq.user_msg.append("H7H Order Subscribed")

    rq.symbol = symbol
    rq.exchange = exchange
    rq.request = exchange_order_notification_pb2.ExchangeOrderNotification.Request.SUBSCRIBE

async def login(ws, system_name, user_id, password):
    rq = request_login_pb2.RequestLogin()

    infra_type = request_login_pb2.RequestLogin.SysInfraType.TICKER_PLANT

    rq.template_id = 10;
    rq.template_version = "3.9"
    rq.user_msg.append("H7H Login")
    rq.user = user_id
    rq.password = password
    rq.app_name = "H7H.py"
    rq.app_version = "0.0.0"
    rq.system_name = system_name
    rq.infra_type = infra_type

    serialized  = rq.SerializeToString()
    length      = len(serialized)

    buf = bytearray()
    buf = serialized

    await ws.send(buf)

    rp_buf = bytearray()
    rp_buf = await ws.recv()

    rp = response_login_pb2.ResponseLogin()
    rp.ParseFromString(rp_buf[0:])

    print(f"      ResponseLogin :")
    print(f"      ===============")
    print(f"        template_id : {rp.template_id}")
    print(f"   template_version : {rp.template_version}")
    print(f"           user_msg : {rp.user_msg}")
    print(f"            rp code : {rp.rp_code}")
    print(f"             fcm_id : {rp.fcm_id}")
    print(f"             ib_id  : {rp.ib_id}")
    print(f"       country_code : {rp.country_code}")
    print(f"         state_code : {rp.state_code}")
    print(f" heartbeat_interval : {rp.heartbeat_interval}")
    print(f"     unique_user_id : {rp.unique_user_id}")

async def logout(ws):
    rq = request_logout_pb2.RequestLogout()

    rq.template_id      = 12;
    rq.user_msg.append("H7H Logout")

    serialized = rq.SerializeToString()

    buf = bytearray()
    buf = serialized

    await ws.send(buf)

async def send_heartbeat(ws):
    rq = request_heartbeat_pb2.RequestHeartbeat()
    rq.template_id = 18

    serialized = rq.SerializeToString()

    buf  = bytearray()
    buf = serialized

    await ws.send(buf)
    print(f"sent heartbeat request")

async def consume(ws, type):
        msg_buf = bytearray()

        waiting_for_msg = True

        while waiting_for_msg:
            try:
                msg_buf = await asyncio.wait_for(ws.recv(), timeout=5)
                waiting_for_msg = False
            except asyncio.TimeoutError:
                if ws.open:
                    await send_heartbeat(ws)
                else:
                    print(f"connection appears to be closed.")
        
        base = base_pb2.Base()

        base.ParseFromString(msg_buf[0:])

        if type == "Last Trade":
            msg = last_trade_pb2.LastTrade()

            msg.ParseFromString(msg_buf[0:])

            return msg
        
        elif type == "BBO":
            msg = best_bid_offer_pb2.BestBidOffer()

            msg.ParseFromString(msg_buf[0:])

            is_bid = True if msg.presence_bits & best_bid_offer_pb2.BestBidOffer.PresenceBits.BID else False
            is_ask = True if msg.presence_bits & best_bid_offer_pb2.BestBidOffer.PresenceBits.ASK else False

            return msg, is_bid, is_ask
            
        elif base.template_id == 13:
            msg_type = "logout response"
            print(f" consumed msg : {msg_type} ({base.template_id})")

        elif base.template_id == 19:
            msg_type = "heartbeat response"
            print(f" consumed msg : {msg_type} ({base.template_id})")

        elif base.template_id == 101:
            msg = response_market_data_update_pb2.ResponseMarketDataUpdate()
            
            msg.ParseFromString(msg_buf[0:])
            print(f"")
            print(f" ResponseMarketDataUpdate : ")
            print(f"                 user_msg :  {msg.user_msg}")
            print(f"                  rp_code :  {msg.rp_code}")

async def disconnect(ws):
    await ws.close(1000, "see you tomorrow")