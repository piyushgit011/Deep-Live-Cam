import asyncio
import logging
from aiortc import RTCPeerConnection
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class WebRTCHandler:
    """Manages WebRTC peer connections"""
    
    def __init__(self):
        self.connections: Dict[str, RTCPeerConnection] = {}
    
    def add_connection(self, connection_id: str, pc: RTCPeerConnection):
        """Add a new WebRTC connection"""
        self.connections[connection_id] = pc
        logger.info(f"Added WebRTC connection: {connection_id}")
    
    def get_connection(self, connection_id: str) -> Optional[RTCPeerConnection]:
        """Get WebRTC connection by ID"""
        return self.connections.get(connection_id)
    
    def remove_connection(self, connection_id: str):
        """Remove WebRTC connection"""
        if connection_id in self.connections:
            del self.connections[connection_id]
            logger.info(f"Removed WebRTC connection: {connection_id}")
    
    async def close_all_connections(self):
        """Close all WebRTC connections"""
        for connection_id, pc in list(self.connections.items()):
            try:
                await pc.close()
            except Exception as e:
                logger.error(f"Error closing connection {connection_id}: {e}")
        
        self.connections.clear()
        logger.info("Closed all WebRTC connections")