from scapy.all import PcapReader
from mypyflowmeter.sniffer import create_sniffer
import time
from mypyflowmeter.flow_session import generate_session_class,FlowSession
import sys
from config import Config




# Function to process each packet
# def packet_callback(packet:Packet):


    # print(packet.fields)
    # print(packet.layers())
    # print(packet.summary())  # Print a summary of the packet
    # Find if packet has syn layer
    # if packet.haslayer('TCP') and packet['TCP'].flags == 'S':
    #     print("SYN packet detected!")
    # if packet.haslayer('IP'):
    #     print(f"Source IP: {packet['IP'].src} -> Destination IP: {packet['IP'].dst}")
    # print()


# # Sniff packets on the network
# def start_sniffing(interface=None, packet_count=10):
#     """
#     Sniffs packets on the given interface.

#     :param interface: The network interface to sniff on. Defaults to None (sniffs all interfaces).
#     :param packet_count: The number of packets to capture. Default is 10.
#     """
#     print(f"Starting to sniff on interface: {interface or 'all interfaces'}...\n")
#     sniff(iface=interface, prn=packet_callback, count=packet_count)
#     print("\nSniffing finished.")

# if __name__ == "__main__":
#     # Replace 'eth0' with your desired interface or use None for all interfaces
#     interface = 'Wi-Fi'  # Example: 'eth0'
#     start_sniffing(interface=interface, packet_count=10)
# from pyflowmeter.sniffer import create_sniffer

# sniffer = create_sniffer(
#             input_file='ipv4frags.pcap',
#             to_csv=True,
#             output_file='./flows_test.csv',
#         )

# sniffer.start()
# try:
#     sniffer.join()
# except KeyboardInterrupt:
#     print('Stopping the sniffer')
#     sniffer.stop()
# finally:
#     sniffer.join()

def test_sniff():
    
    sniffer = create_sniffer(
        input_file='mycapture.pcapng',
        server_endpoint='http://localhost:5000/flows',
        verbose=True,
    )

    sniffer.start()
    try:
        sniffer.join()
    except KeyboardInterrupt:
        print('Stopping the sniffer')
        sniffer.stop()
    finally:
        sniffer.join()

def test_offline():
    file='mycapture.pcapng'
    NewFlowSession = generate_session_class('http://localhost:5000/flows', True, False, None, 1)
    reader = PcapReader(file)
    packets=[packet for packet in reader]
    session = NewFlowSession()
    for packet in packets:
        session.on_packet_received(packet)
    session.toPacketList()



def simulate():
    """
    Entry point for Intrusion Detection System. It reads the pcap file and simulates the flow of packets.

    """
    config = Config()
    file=config.get_data(sys.argv[1])
    NewFlowSession = generate_session_class('http://localhost:5000/flows', False, False, None, 1)
    reader = PcapReader(file)
    print("Starting simulation")
    session = NewFlowSession()
    firstPacketTime=None
    startTime=time.time()
    flag=False
    for packet in reader:
        if not flag:
            firstPacketTime=float(packet.time)
            flag=True
        currentTime=time.time()
        packetRelativeTime=float(packet.time)-firstPacketTime
        if currentTime-startTime<packetRelativeTime:
            time.sleep(packetRelativeTime-(currentTime-startTime))
        session.on_packet_received(packet)
    session.toPacketList()


simulate()