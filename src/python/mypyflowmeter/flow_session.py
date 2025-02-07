import time
from threading import Thread, Lock
import csv

from scapy.sessions import DefaultSession
from scapy.packet import Packet
from .features.context.packet_direction import PacketDirection
from .features.context.packet_flow_key import get_packet_flow_key
from .flow import Flow
from dt import tree,detect_intrusion


EXPIRED_UPDATE = 40
SENDING_INTERVAL = 1

class Host:
    def __init__(self,ip,port):
        self.ip=ip
        self.port=port
        self.threat='BENIGN'
    
    def __str__(self):
        return self.ip+":"+str(self.port)+" - Threat:"+self.threat
    
    def __hash__(self):
        return hash(self.ip+str(self.port))
    
    def __eq__(self,other):
        return self.ip==other.ip and self.port==other.port

class HostTracer:
    def __init__(self):
        self.hosts=set()

    def addHost(self,ip,port):
        host=Host(ip,port)
        self.hosts.add(host)
        return host
    
    def getHost(self,ip,port):
        host=Host(ip,port)
        for h in self.hosts:
            if h==host:
                return h
        return None
    
    def setThreat(self,ip,port,threat):
        host=self.getHost(ip,port)
        if host is not None:
            host.threat=threat
        else:
            host=self.addHost(ip,port)
            host.threat=threat

hosttracer=HostTracer()

class FlowSession(DefaultSession):
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.csv_line = 0
        self.packets_count = 0
        self.GARBAGE_COLLECT_PACKETS = 10000 if self.server_endpoint is None else 100
        self.finished = False
        print(self.server_endpoint)
        self.lock = Lock() 
        if self.server_endpoint is not None:
            thread = Thread(target=self.send_flows_to_server)
            thread.start()
            self.thread = thread
        if self.to_csv:
            output = open(self.output_file, "w")
            self.csv_writer = csv.writer(output)

        super(FlowSession, self).__init__(*args, **kwargs)

    def processFlows(self):
        if len(self.flows) != 0:
            with self.lock:
                flows = list(self.flows.values())
            self.garbage_collect()
            data = {'flows': [flow.get_data() for flow in flows]}
            print("processing {} flows".format(len(flows)))
            start=time.time()
            predictions=detect_intrusion(data)
            # print("Time taken for prediction:",time.time()-start)
            for i in range(len(predictions)):
                flow=flows[i]
                prediction=predictions[i]
                hosttracer.addHost(flow.src_ip,flow.src_port)
                if prediction.strip()!="BENIGN":
                    print("Threat detected in flow",prediction)
                    hosttracer.setThreat(flow.src_ip,flow.src_port,prediction)
            #requests.post(self.server_endpoint, json=data)

    def send_flows_to_server(self):
        while not self.finished:
            self.processFlows()
            time.sleep(self.sending_interval)
        self.processFlows()
        for host in hosttracer.hosts:
            print(host)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.finished = True

        # return super(FlowSession, self).toPacketList()
    

    def on_packet_received(self, packet):
        count = 0
        direction = PacketDirection.FORWARD
        try:
            # Creates a key variable to check
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception as e:
            return

        self.packets_count += 1
        if self.verbose:
            print('New packet received. Count: ' + str(self.packets_count))

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

        if flow is None:
            # print("creating flow")
            # If no flow exists create a new flow
            direction = PacketDirection.FORWARD
            flow = Flow(packet, direction)
            packet_flow_key = get_packet_flow_key(packet, direction)
            with self.lock:
                self.flows[(packet_flow_key, count)] = flow

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            # If the packet exists in the flow but the packet is sent
            # after too much of a delay than it is a part of a new flow.
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    with self.lock:
                        self.flows[(packet_flow_key, count)] = flow
                    break
        elif "F" in str(packet.flags):
            # If it has FIN flag then early collect flow and continue
            flow.add_packet(packet, direction)
            # self.garbage_collect(packet.time)                    
            return

        flow.add_packet(packet, direction)

        if self.packets_count % self.GARBAGE_COLLECT_PACKETS == 0 or (
            flow.duration > 120 
        ):
            self.garbage_collect()
        return packet

    def get_flows(self) -> list:
        return self.flows.values()
    
    def write_data_csv(self):
        with self.lock:
            flows = list(self.flows.values())
        for flow in flows:
            data = flow.get_data()

            if self.csv_line == 0:
                self.csv_writer.writerow(data.keys())

            self.csv_writer.writerow(data.values())
            self.csv_line += 1

    def garbage_collect(self) -> None:
        if self.to_csv:
            self.write_data_csv()
        with self.lock:
            self.flows = {}


def generate_session_class(server_endpoint=None, verbose=False, to_csv=False, output_file=None, sending_interval=1):
    return type(
        "NewFlowSession",
        (FlowSession,),
        {
            "server_endpoint": server_endpoint,
            "verbose": verbose,
            "to_csv": to_csv,
            "output_file": output_file,
            "sending_interval": sending_interval
        },
    )
