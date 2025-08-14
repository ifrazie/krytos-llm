from typing import Any, Dict, Optional, List
import platform
import os
import socket
import psutil
import time
from datetime import datetime, timedelta
import httpx
import asyncio
from mcp.server.fastmcp import FastMCP


class SystemInfoCollector:
    """Collector class for basic system information including OS, hostname, user, and uptime."""
    
    @staticmethod
    def get_os_info() -> Dict[str, str]:
        """Get operating system information.
        
        Returns:
            Dict containing OS name, version, and architecture information.
        """
        try:
            return {
                'name': platform.system(),
                'version': platform.version(),
                'release': platform.release(),
                'architecture': platform.machine()
            }
        except Exception as e:
            return {
                'name': 'Unknown',
                'version': 'Unknown',
                'release': 'Unknown',
                'architecture': 'Unknown',
                'error': str(e)
            }
    
    @staticmethod
    def get_hostname_info() -> Dict[str, str]:
        """Get hostname and domain information.
        
        Returns:
            Dict containing hostname and domain information.
        """
        try:
            hostname = socket.gethostname()
            try:
                # Try to get FQDN
                fqdn = socket.getfqdn()
                if '.' in fqdn and fqdn != hostname:
                    domain = fqdn.split('.', 1)[1]
                else:
                    domain = 'Unknown'
            except Exception:
                domain = 'Unknown'
            
            return {
                'hostname': hostname,
                'domain': domain,
                'fqdn': fqdn if 'fqdn' in locals() else hostname
            }
        except Exception as e:
            return {
                'hostname': 'Unknown',
                'domain': 'Unknown',
                'fqdn': 'Unknown',
                'error': str(e)
            }
    
    @staticmethod
    def get_user_info() -> Dict[str, str]:
        """Get current user information.
        
        Returns:
            Dict containing current user information.
        """
        try:
            # Try to get username
            username = 'Unknown'
            if hasattr(os, 'getlogin'):
                try:
                    username = os.getlogin()
                except (OSError, AttributeError):
                    # Fallback to environment variables
                    username = os.environ.get('USER', os.environ.get('USERNAME', 'Unknown'))
            else:
                username = os.environ.get('USER', os.environ.get('USERNAME', 'Unknown'))
            
            user_info = {
                'username': username,
                'uid': str(os.getuid()) if hasattr(os, 'getuid') else 'Unknown',
                'gid': str(os.getgid()) if hasattr(os, 'getgid') else 'Unknown'
            }
            
            return user_info
        except Exception as e:
            return {
                'username': os.environ.get('USER', os.environ.get('USERNAME', 'Unknown')),
                'uid': 'Unknown',
                'gid': 'Unknown',
                'error': str(e)
            }
    
    @staticmethod
    def get_uptime() -> str:
        """Get system uptime information.
        
        Returns:
            String representation of system uptime.
        """
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_delta = timedelta(seconds=int(uptime_seconds))
            
            # Format uptime as "X days, Y hours, Z minutes"
            days = uptime_delta.days
            hours, remainder = divmod(uptime_delta.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            
            if days > 0:
                return f"{days} days, {hours:02d}:{minutes:02d}:00"
            else:
                return f"{hours:02d}:{minutes:02d}:00"
                
        except Exception as e:
            return f"Unknown (Error: {str(e)})"


class HardwareInfoCollector:
    """Collector class for hardware information including CPU, memory, and disk details."""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information including model, cores, and architecture.
        
        Returns:
            Dict containing CPU details including model, physical cores, logical cores, and frequency.
        """
        try:
            cpu_info = {
                'model': 'Unknown',
                'physical_cores': 0,
                'logical_cores': 0,
                'max_frequency': 0.0,
                'current_frequency': 0.0,
                'architecture': platform.machine()
            }
            
            # Get CPU model/brand
            try:
                # Try to get CPU brand string
                if hasattr(psutil, 'cpu_freq') and psutil.cpu_freq():
                    cpu_info['max_frequency'] = psutil.cpu_freq().max
                    cpu_info['current_frequency'] = psutil.cpu_freq().current
                
                # Get core counts
                cpu_info['physical_cores'] = psutil.cpu_count(logical=False) or 0
                cpu_info['logical_cores'] = psutil.cpu_count(logical=True) or 0
                
                # Try to get CPU model from platform
                processor = platform.processor()
                if processor and processor.strip():
                    cpu_info['model'] = processor.strip()
                else:
                    # Fallback to basic platform info
                    cpu_info['model'] = f"{platform.machine()} processor"
                    
            except Exception as e:
                cpu_info['error'] = f"Error getting CPU details: {str(e)}"
            
            return cpu_info
            
        except Exception as e:
            return {
                'model': 'Unknown',
                'physical_cores': 0,
                'logical_cores': 0,
                'max_frequency': 0.0,
                'current_frequency': 0.0,
                'architecture': 'Unknown',
                'error': str(e)
            }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory information including total and available memory.
        
        Returns:
            Dict containing memory statistics in bytes and formatted strings.
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_info = {
                'total_bytes': memory.total,
                'available_bytes': memory.available,
                'used_bytes': memory.used,
                'free_bytes': memory.free,
                'percent_used': memory.percent,
                'total_formatted': HardwareInfoCollector._format_bytes(memory.total),
                'available_formatted': HardwareInfoCollector._format_bytes(memory.available),
                'used_formatted': HardwareInfoCollector._format_bytes(memory.used),
                'swap_total_bytes': swap.total,
                'swap_used_bytes': swap.used,
                'swap_free_bytes': swap.free,
                'swap_percent': swap.percent,
                'swap_total_formatted': HardwareInfoCollector._format_bytes(swap.total),
                'swap_used_formatted': HardwareInfoCollector._format_bytes(swap.used)
            }
            
            return memory_info
            
        except Exception as e:
            return {
                'total_bytes': 0,
                'available_bytes': 0,
                'used_bytes': 0,
                'free_bytes': 0,
                'percent_used': 0.0,
                'total_formatted': '0 B',
                'available_formatted': '0 B',
                'used_formatted': '0 B',
                'swap_total_bytes': 0,
                'swap_used_bytes': 0,
                'swap_free_bytes': 0,
                'swap_percent': 0.0,
                'swap_total_formatted': '0 B',
                'swap_used_formatted': '0 B',
                'error': str(e)
            }
    
    @staticmethod
    def get_disk_info() -> List[Dict[str, Any]]:
        """Get disk space information for all mounted drives.
        
        Returns:
            List of dicts containing disk information for each mounted drive.
        """
        try:
            disks = []
            disk_partitions = psutil.disk_partitions()
            
            for partition in disk_partitions:
                try:
                    # Skip special filesystems and network drives
                    if HardwareInfoCollector._should_skip_partition(partition):
                        continue
                    
                    # Get disk usage for this partition
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    disk_info = {
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'filesystem': partition.fstype,
                        'total_bytes': usage.total,
                        'used_bytes': usage.used,
                        'free_bytes': usage.free,
                        'percent_used': (usage.used / usage.total * 100) if usage.total > 0 else 0,
                        'total_formatted': HardwareInfoCollector._format_bytes(usage.total),
                        'used_formatted': HardwareInfoCollector._format_bytes(usage.used),
                        'free_formatted': HardwareInfoCollector._format_bytes(usage.free),
                        'is_primary': HardwareInfoCollector._is_primary_disk(partition)
                    }
                    
                    disks.append(disk_info)
                    
                except (PermissionError, OSError) as e:
                    # Skip partitions we can't access
                    continue
            
            # Sort disks with primary disk first, then by mountpoint
            disks.sort(key=lambda x: (not x['is_primary'], x['mountpoint']))
            
            return disks
            
        except Exception as e:
            return [{
                'device': 'Unknown',
                'mountpoint': 'Unknown',
                'filesystem': 'Unknown',
                'total_bytes': 0,
                'used_bytes': 0,
                'free_bytes': 0,
                'percent_used': 0.0,
                'total_formatted': '0 B',
                'used_formatted': '0 B',
                'free_formatted': '0 B',
                'is_primary': False,
                'error': str(e)
            }]
    
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes into human-readable string.
        
        Args:
            bytes_value: Number of bytes to format
            
        Returns:
            Formatted string (e.g., "1.5 GB", "512 MB")
        """
        if bytes_value == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0
        size = float(bytes_value)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.1f} {units[unit_index]}"
    
    @staticmethod
    def _should_skip_partition(partition) -> bool:
        """Determine if a partition should be skipped based on filesystem type and mountpoint.
        
        Args:
            partition: psutil disk partition object
            
        Returns:
            True if partition should be skipped, False otherwise
        """
        # Skip common virtual/special filesystems
        skip_fstypes = {
            'proc', 'sysfs', 'devfs', 'devtmpfs', 'tmpfs', 'cgroup', 'cgroup2',
            'pstore', 'bpf', 'debugfs', 'tracefs', 'securityfs', 'hugetlbfs',
            'mqueue', 'configfs', 'fusectl', 'selinuxfs', 'overlay', 'squashfs'
        }
        
        if partition.fstype.lower() in skip_fstypes:
            return True
        
        # Skip network filesystems
        network_fstypes = {'nfs', 'nfs4', 'cifs', 'smb', 'smbfs', 'ftp', 'sftp'}
        if partition.fstype.lower() in network_fstypes:
            return True
        
        # Skip snap mounts on Linux
        if '/snap/' in partition.mountpoint:
            return True
        
        # Skip common virtual mountpoints
        virtual_mounts = {'/proc', '/sys', '/dev', '/run', '/tmp'}
        if partition.mountpoint in virtual_mounts:
            return True
        
        return False
    
    @staticmethod
    def _is_primary_disk(partition) -> bool:
        """Determine if a partition is the primary/system disk.
        
        Args:
            partition: psutil disk partition object
            
        Returns:
            True if this is likely the primary disk, False otherwise
        """
        # On Windows, C: is typically primary
        if platform.system() == 'Windows':
            return partition.mountpoint.upper().startswith('C:')
        
        # On Unix-like systems, / is the root filesystem
        return partition.mountpoint == '/'


class NetworkInfoCollector:
    """Collector class for network information including interfaces, gateway, DNS, and public IP."""
    
    @staticmethod
    def get_interfaces() -> List[Dict[str, Any]]:
        """Get information about active network interfaces.
        
        Returns:
            List of dicts containing interface information including name, IP addresses, and MAC addresses.
        """
        try:
            interfaces = []
            network_interfaces = psutil.net_if_addrs()
            network_stats = psutil.net_if_stats()
            
            for interface_name, addresses in network_interfaces.items():
                # Skip loopback and inactive interfaces
                if NetworkInfoCollector._should_skip_interface(interface_name, network_stats):
                    continue
                
                interface_info = {
                    'name': interface_name,
                    'ip_addresses': [],
                    'mac_address': None,
                    'is_active': False,
                    'is_up': False,
                    'speed': 0,
                    'mtu': 0
                }
                
                # Get interface statistics if available
                if interface_name in network_stats:
                    stats = network_stats[interface_name]
                    interface_info['is_up'] = stats.isup
                    interface_info['speed'] = stats.speed if stats.speed != -1 else 0
                    interface_info['mtu'] = stats.mtu
                
                # Process addresses for this interface
                for addr in addresses:
                    if addr.family == socket.AF_INET:  # IPv4
                        ip_info = {
                            'ip': addr.address,
                            'netmask': addr.netmask,
                            'broadcast': addr.broadcast,
                            'family': 'IPv4'
                        }
                        interface_info['ip_addresses'].append(ip_info)
                        # Mark as active if it has a non-loopback IPv4 address
                        if not addr.address.startswith('127.'):
                            interface_info['is_active'] = True
                    elif addr.family == socket.AF_INET6:  # IPv6
                        ip_info = {
                            'ip': addr.address,
                            'netmask': addr.netmask,
                            'broadcast': addr.broadcast,
                            'family': 'IPv6'
                        }
                        interface_info['ip_addresses'].append(ip_info)
                    elif addr.family == psutil.AF_LINK:  # MAC address
                        interface_info['mac_address'] = addr.address
                
                # Only include interfaces with IP addresses
                if interface_info['ip_addresses']:
                    interfaces.append(interface_info)
            
            # Sort interfaces with active ones first
            interfaces.sort(key=lambda x: (not x['is_active'], x['name']))
            
            return interfaces
            
        except Exception as e:
            return [{
                'name': 'Unknown',
                'ip_addresses': [],
                'mac_address': None,
                'is_active': False,
                'is_up': False,
                'speed': 0,
                'mtu': 0,
                'error': str(e)
            }]
    
    @staticmethod
    def get_gateway_info() -> Dict[str, Any]:
        """Get default gateway information.
        
        Returns:
            Dict containing default gateway IP and interface information.
        """
        gateway_info = {
            'gateway_ip': None,
            'interface': None,
            'family': None
        }
        
        try:
            # Method 1: Try to connect to a remote address to determine default route
            try:
                # Create a socket and connect to a remote address (doesn't actually send data)
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    
                    # Find which interface has this IP
                    network_interfaces = psutil.net_if_addrs()
                    for interface_name, addresses in network_interfaces.items():
                        for addr in addresses:
                            if addr.family == socket.AF_INET and addr.address == local_ip:
                                gateway_info['interface'] = interface_name
                                gateway_info['family'] = 'IPv4'
                                
                                # Try to determine gateway IP from network
                                try:
                                    # Parse network to guess gateway (usually .1 in the subnet)
                                    ip_parts = local_ip.split('.')
                                    if len(ip_parts) == 4:
                                        # Common gateway patterns
                                        if ip_parts[0] == '192' and ip_parts[1] == '168':
                                            gateway_info['gateway_ip'] = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.1"
                                        elif ip_parts[0] == '10':
                                            gateway_info['gateway_ip'] = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.1"
                                        elif ip_parts[0] == '172' and 16 <= int(ip_parts[1]) <= 31:
                                            gateway_info['gateway_ip'] = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.1"
                                except Exception:
                                    pass
                                break
                        if gateway_info['interface']:
                            break
            except Exception:
                pass
            
            # Method 2: Platform-specific approaches
            if not gateway_info['gateway_ip']:
                try:
                    if platform.system() == 'Windows':
                        # On Windows, try to parse route table
                        import subprocess
                        result = subprocess.run(['route', 'print', '0.0.0.0'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            lines = result.stdout.split('\n')
                            for line in lines:
                                if '0.0.0.0' in line and 'Gateway' not in line:
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        gateway_info['gateway_ip'] = parts[2]
                                        break
                    else:
                        # On Unix-like systems, try ip route or route command
                        import subprocess
                        try:
                            result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                                  capture_output=True, text=True, timeout=5)
                            if result.returncode == 0:
                                parts = result.stdout.split()
                                if 'via' in parts:
                                    via_index = parts.index('via')
                                    if via_index + 1 < len(parts):
                                        gateway_info['gateway_ip'] = parts[via_index + 1]
                                if 'dev' in parts:
                                    dev_index = parts.index('dev')
                                    if dev_index + 1 < len(parts):
                                        gateway_info['interface'] = parts[dev_index + 1]
                        except (subprocess.SubprocessError, FileNotFoundError):
                            # Try alternative route command
                            try:
                                result = subprocess.run(['route', '-n', 'get', 'default'], 
                                                      capture_output=True, text=True, timeout=5)
                                if result.returncode == 0:
                                    lines = result.stdout.split('\n')
                                    for line in lines:
                                        if 'gateway:' in line.lower():
                                            gateway_info['gateway_ip'] = line.split(':')[1].strip()
                                        elif 'interface:' in line.lower():
                                            gateway_info['interface'] = line.split(':')[1].strip()
                            except (subprocess.SubprocessError, FileNotFoundError):
                                pass
                except Exception:
                    pass
            
        except Exception as e:
            gateway_info['error'] = str(e)
        
        return gateway_info
    
    @staticmethod
    def get_dns_info() -> List[str]:
        """Get DNS server configuration.
        
        Returns:
            List of DNS server IP addresses.
        """
        try:
            dns_servers = []
            
            # Method 1: Try to read system DNS configuration
            if platform.system() == 'Windows':
                # On Windows, try to get DNS from network adapter configuration
                try:
                    import subprocess
                    result = subprocess.run(['nslookup', 'localhost'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'Address:' in line and 'Server:' not in line:
                                # Look for Address line that follows Server line
                                address = line.split(':')[1].strip()
                                if address and address != 'localhost' and address != '127.0.0.1':
                                    # Validate it's an IP address
                                    if NetworkInfoCollector._is_valid_ip(address):
                                        dns_servers.append(address)
                except Exception:
                    pass
            else:
                # On Unix-like systems, try to read /etc/resolv.conf
                try:
                    with open('/etc/resolv.conf', 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('nameserver'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    dns_server = parts[1]
                                    # Skip localhost entries
                                    if dns_server not in ['127.0.0.1', '::1', 'localhost']:
                                        dns_servers.append(dns_server)
                except (FileNotFoundError, PermissionError):
                    pass
            
            # Method 2: Try using socket.getaddrinfo to detect DNS
            if not dns_servers:
                try:
                    # This is indirect - we can't directly get DNS servers from socket module
                    # But we can try some common DNS servers and see if they're reachable
                    common_dns = ['8.8.8.8', '8.8.4.4', '1.1.1.1', '1.0.0.1']
                    for dns in common_dns:
                        try:
                            socket.create_connection((dns, 53), timeout=1).close()
                            dns_servers.append(dns)
                            break  # Just add one working DNS server
                        except (socket.timeout, socket.error):
                            continue
                except Exception:
                    pass
            
            # Remove duplicates while preserving order
            seen = set()
            unique_dns = []
            for dns in dns_servers:
                if dns not in seen:
                    seen.add(dns)
                    unique_dns.append(dns)
            
            return unique_dns[:4]  # Limit to first 4 DNS servers
            
        except Exception as e:
            return []
    
    @staticmethod
    async def get_public_ip() -> str:
        """Get public IP address using external service.
        
        Returns:
            String containing public IP address or error message.
        """
        try:
            # List of public IP services to try (following weather.py pattern)
            ip_services = [
                "https://api.ipify.org",
                "https://ipv4.icanhazip.com",
                "https://checkip.amazonaws.com",
                "https://ipinfo.io/ip"
            ]
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                for service_url in ip_services:
                    try:
                        response = await client.get(service_url)
                        if response.status_code == 200:
                            public_ip = response.text.strip()
                            # Basic validation - check if it looks like an IP address
                            if NetworkInfoCollector._is_valid_ip(public_ip):
                                return public_ip
                    except (httpx.RequestError, httpx.TimeoutException):
                        continue
                
                # If all services failed
                return "Unable to determine public IP"
                
        except Exception as e:
            return f"Error getting public IP: {str(e)}"
    
    @staticmethod
    def _should_skip_interface(interface_name: str, network_stats: Dict) -> bool:
        """Determine if a network interface should be skipped.
        
        Args:
            interface_name: Name of the network interface
            network_stats: Dictionary of network interface statistics
            
        Returns:
            True if interface should be skipped, False otherwise
        """
        # Skip loopback interfaces
        if interface_name.lower().startswith(('lo', 'loopback')):
            return True
        
        # Skip virtual interfaces (common patterns)
        virtual_patterns = [
            'veth', 'docker', 'br-', 'virbr', 'vmnet', 'vboxnet',
            'tun', 'tap', 'ppp', 'wwan', 'isatap', 'teredo'
        ]
        
        interface_lower = interface_name.lower()
        for pattern in virtual_patterns:
            if pattern in interface_lower:
                return True
        
        # Skip interfaces that are down (if stats available)
        if interface_name in network_stats:
            stats = network_stats[interface_name]
            if not stats.isup:
                return True
        
        return False
    
    @staticmethod
    def _is_valid_ip(ip_string: str) -> bool:
        """Validate if a string is a valid IP address.
        
        Args:
            ip_string: String to validate as IP address
            
        Returns:
            True if valid IP address, False otherwise
        """
        if not ip_string or not isinstance(ip_string, str):
            return False
            
        try:
            # Try to parse as IPv4 - inet_aton is more strict than inet_pton
            parts = ip_string.split('.')
            if len(parts) == 4:
                for part in parts:
                    if not part.isdigit() or not (0 <= int(part) <= 255):
                        return False
                socket.inet_aton(ip_string)
                return True
        except (socket.error, ValueError):
            pass
            
        try:
            # Try to parse as IPv6
            socket.inet_pton(socket.AF_INET6, ip_string)
            return True
        except (socket.error, OSError):
            return False
        
        return False


# Initialize FastMCP server
mcp = FastMCP("host-details")


class ResourceUsageCollector:
    """Collector class for real-time system resource usage including CPU, memory, disk I/O, and network I/O."""
    
    @staticmethod
    def get_cpu_usage(interval: float = 1.0) -> Dict[str, Any]:
        """Get current CPU usage statistics.
        
        Args:
            interval: Time interval in seconds for CPU usage measurement (default: 1.0)
            
        Returns:
            Dict containing CPU usage information including overall and per-core usage.
        """
        try:
            # Get overall CPU usage with proper sampling interval
            cpu_percent = psutil.cpu_percent(interval=interval)
            
            # Get per-core CPU usage (non-blocking after initial call)
            per_cpu_percent = psutil.cpu_percent(interval=0, percpu=True)
            
            # Get CPU times for more detailed statistics
            cpu_times = psutil.cpu_times()
            
            # Get CPU frequency information
            cpu_freq = None
            try:
                cpu_freq = psutil.cpu_freq()
            except (AttributeError, OSError):
                pass
            
            cpu_info = {
                'overall_percent': round(cpu_percent, 1),
                'per_core_percent': [round(core, 1) for core in per_cpu_percent],
                'core_count': len(per_cpu_percent),
                'times': {
                    'user': getattr(cpu_times, 'user', 0),
                    'system': getattr(cpu_times, 'system', 0),
                    'idle': getattr(cpu_times, 'idle', 0),
                    'iowait': getattr(cpu_times, 'iowait', 0) if hasattr(cpu_times, 'iowait') else 0,
                    'irq': getattr(cpu_times, 'irq', 0) if hasattr(cpu_times, 'irq') else 0,
                    'softirq': getattr(cpu_times, 'softirq', 0) if hasattr(cpu_times, 'softirq') else 0
                },
                'frequency': {
                    'current': cpu_freq.current if cpu_freq else 0,
                    'min': cpu_freq.min if cpu_freq else 0,
                    'max': cpu_freq.max if cpu_freq else 0
                } if cpu_freq else None,
                'load_average': ResourceUsageCollector._get_load_average()
            }
            
            return cpu_info
            
        except Exception as e:
            return {
                'overall_percent': 0.0,
                'per_core_percent': [],
                'core_count': 0,
                'times': {
                    'user': 0, 'system': 0, 'idle': 0, 'iowait': 0, 'irq': 0, 'softirq': 0
                },
                'frequency': None,
                'load_average': None,
                'error': str(e)
            }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get current memory usage statistics.
        
        Returns:
            Dict containing memory usage information including virtual and swap memory.
        """
        try:
            # Get virtual memory statistics
            virtual_memory = psutil.virtual_memory()
            
            # Get swap memory statistics
            swap_memory = psutil.swap_memory()
            
            memory_info = {
                'virtual': {
                    'total_bytes': virtual_memory.total,
                    'available_bytes': virtual_memory.available,
                    'used_bytes': virtual_memory.used,
                    'free_bytes': virtual_memory.free,
                    'percent_used': round(virtual_memory.percent, 1),
                    'total_formatted': ResourceUsageCollector._format_bytes(virtual_memory.total),
                    'available_formatted': ResourceUsageCollector._format_bytes(virtual_memory.available),
                    'used_formatted': ResourceUsageCollector._format_bytes(virtual_memory.used),
                    'free_formatted': ResourceUsageCollector._format_bytes(virtual_memory.free),
                    'buffers': getattr(virtual_memory, 'buffers', 0),
                    'cached': getattr(virtual_memory, 'cached', 0),
                    'shared': getattr(virtual_memory, 'shared', 0) if hasattr(virtual_memory, 'shared') else 0
                },
                'swap': {
                    'total_bytes': swap_memory.total,
                    'used_bytes': swap_memory.used,
                    'free_bytes': swap_memory.free,
                    'percent_used': round(swap_memory.percent, 1),
                    'total_formatted': ResourceUsageCollector._format_bytes(swap_memory.total),
                    'used_formatted': ResourceUsageCollector._format_bytes(swap_memory.used),
                    'free_formatted': ResourceUsageCollector._format_bytes(swap_memory.free),
                    'sin': getattr(swap_memory, 'sin', 0),  # bytes swapped in
                    'sout': getattr(swap_memory, 'sout', 0)  # bytes swapped out
                }
            }
            
            return memory_info
            
        except Exception as e:
            return {
                'virtual': {
                    'total_bytes': 0, 'available_bytes': 0, 'used_bytes': 0, 'free_bytes': 0,
                    'percent_used': 0.0, 'total_formatted': '0 B', 'available_formatted': '0 B',
                    'used_formatted': '0 B', 'free_formatted': '0 B',
                    'buffers': 0, 'cached': 0, 'shared': 0
                },
                'swap': {
                    'total_bytes': 0, 'used_bytes': 0, 'free_bytes': 0, 'percent_used': 0.0,
                    'total_formatted': '0 B', 'used_formatted': '0 B', 'free_formatted': '0 B',
                    'sin': 0, 'sout': 0
                },
                'error': str(e)
            }
    
    @staticmethod
    def get_disk_io() -> Dict[str, Any]:
        """Get disk I/O statistics.
        
        Returns:
            Dict containing disk I/O information including read/write statistics per disk and totals.
        """
        try:
            # Get disk I/O counters
            disk_io = psutil.disk_io_counters(perdisk=True)
            disk_io_total = psutil.disk_io_counters(perdisk=False)
            
            # Process per-disk statistics
            per_disk_stats = {}
            if disk_io:
                for disk_name, io_counters in disk_io.items():
                    # Skip virtual/loop devices on Linux
                    if ResourceUsageCollector._should_skip_disk(disk_name):
                        continue
                    
                    per_disk_stats[disk_name] = {
                        'read_count': io_counters.read_count,
                        'write_count': io_counters.write_count,
                        'read_bytes': io_counters.read_bytes,
                        'write_bytes': io_counters.write_bytes,
                        'read_time': io_counters.read_time,
                        'write_time': io_counters.write_time,
                        'read_bytes_formatted': ResourceUsageCollector._format_bytes(io_counters.read_bytes),
                        'write_bytes_formatted': ResourceUsageCollector._format_bytes(io_counters.write_bytes),
                        'busy_time': getattr(io_counters, 'busy_time', 0) if hasattr(io_counters, 'busy_time') else 0
                    }
            
            # Total statistics
            total_stats = {}
            if disk_io_total:
                total_stats = {
                    'read_count': disk_io_total.read_count,
                    'write_count': disk_io_total.write_count,
                    'read_bytes': disk_io_total.read_bytes,
                    'write_bytes': disk_io_total.write_bytes,
                    'read_time': disk_io_total.read_time,
                    'write_time': disk_io_total.write_time,
                    'read_bytes_formatted': ResourceUsageCollector._format_bytes(disk_io_total.read_bytes),
                    'write_bytes_formatted': ResourceUsageCollector._format_bytes(disk_io_total.write_bytes),
                    'busy_time': getattr(disk_io_total, 'busy_time', 0) if hasattr(disk_io_total, 'busy_time') else 0
                }
            
            disk_info = {
                'per_disk': per_disk_stats,
                'total': total_stats,
                'disk_count': len(per_disk_stats)
            }
            
            return disk_info
            
        except Exception as e:
            return {
                'per_disk': {},
                'total': {
                    'read_count': 0, 'write_count': 0, 'read_bytes': 0, 'write_bytes': 0,
                    'read_time': 0, 'write_time': 0, 'read_bytes_formatted': '0 B',
                    'write_bytes_formatted': '0 B', 'busy_time': 0
                },
                'disk_count': 0,
                'error': str(e)
            }
    
    @staticmethod
    def get_network_io() -> Dict[str, Any]:
        """Get network I/O statistics.
        
        Returns:
            Dict containing network I/O information including bytes sent/received per interface and totals.
        """
        try:
            # Get network I/O counters
            network_io = psutil.net_io_counters(pernic=True)
            network_io_total = psutil.net_io_counters(pernic=False)
            
            # Process per-interface statistics
            per_interface_stats = {}
            if network_io:
                for interface_name, io_counters in network_io.items():
                    # Skip loopback and virtual interfaces
                    if ResourceUsageCollector._should_skip_network_interface(interface_name):
                        continue
                    
                    per_interface_stats[interface_name] = {
                        'bytes_sent': io_counters.bytes_sent,
                        'bytes_recv': io_counters.bytes_recv,
                        'packets_sent': io_counters.packets_sent,
                        'packets_recv': io_counters.packets_recv,
                        'errin': io_counters.errin,
                        'errout': io_counters.errout,
                        'dropin': io_counters.dropin,
                        'dropout': io_counters.dropout,
                        'bytes_sent_formatted': ResourceUsageCollector._format_bytes(io_counters.bytes_sent),
                        'bytes_recv_formatted': ResourceUsageCollector._format_bytes(io_counters.bytes_recv)
                    }
            
            # Total statistics
            total_stats = {}
            if network_io_total:
                total_stats = {
                    'bytes_sent': network_io_total.bytes_sent,
                    'bytes_recv': network_io_total.bytes_recv,
                    'packets_sent': network_io_total.packets_sent,
                    'packets_recv': network_io_total.packets_recv,
                    'errin': network_io_total.errin,
                    'errout': network_io_total.errout,
                    'dropin': network_io_total.dropin,
                    'dropout': network_io_total.dropout,
                    'bytes_sent_formatted': ResourceUsageCollector._format_bytes(network_io_total.bytes_sent),
                    'bytes_recv_formatted': ResourceUsageCollector._format_bytes(network_io_total.bytes_recv)
                }
            
            network_info = {
                'per_interface': per_interface_stats,
                'total': total_stats,
                'interface_count': len(per_interface_stats)
            }
            
            return network_info
            
        except Exception as e:
            return {
                'per_interface': {},
                'total': {
                    'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0,
                    'errin': 0, 'errout': 0, 'dropin': 0, 'dropout': 0,
                    'bytes_sent_formatted': '0 B', 'bytes_recv_formatted': '0 B'
                },
                'interface_count': 0,
                'error': str(e)
            }
    
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes into human-readable string.
        
        Args:
            bytes_value: Number of bytes to format
            
        Returns:
            Formatted string (e.g., "1.5 GB", "512 MB")
        """
        if bytes_value == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0
        size = float(bytes_value)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.1f} {units[unit_index]}"
    
    @staticmethod
    def _get_load_average() -> Optional[List[float]]:
        """Get system load average (Unix-like systems only).
        
        Returns:
            List of load averages [1min, 5min, 15min] or None if not available.
        """
        try:
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                return [round(avg, 2) for avg in load_avg]
        except (OSError, AttributeError):
            pass
        return None
    
    @staticmethod
    def _should_skip_disk(disk_name: str) -> bool:
        """Determine if a disk should be skipped from I/O statistics.
        
        Args:
            disk_name: Name of the disk device
            
        Returns:
            True if disk should be skipped, False otherwise
        """
        # Skip loop devices, ram disks, and other virtual devices on Linux
        virtual_disk_patterns = [
            'loop', 'ram', 'dm-', 'md', 'sr', 'fd'
        ]
        
        disk_lower = disk_name.lower()
        for pattern in virtual_disk_patterns:
            if pattern in disk_lower:
                return True
        
        return False
    
    @staticmethod
    def _should_skip_network_interface(interface_name: str) -> bool:
        """Determine if a network interface should be skipped from I/O statistics.
        
        Args:
            interface_name: Name of the network interface
            
        Returns:
            True if interface should be skipped, False otherwise
        """
        # Skip loopback and virtual interfaces
        skip_patterns = [
            'lo', 'loopback', 'veth', 'docker', 'br-', 'virbr', 'vmnet', 'vboxnet',
            'tun', 'tap', 'ppp', 'wwan', 'isatap', 'teredo'
        ]
        
        interface_lower = interface_name.lower()
        for pattern in skip_patterns:
            if pattern in interface_lower:
                return True
        
        return False


class ProcessInfoCollector:
    """Collector class for process and service information including running processes with filtering capabilities."""
    
    @staticmethod
    def get_processes(filter_name: str = "") -> List[Dict[str, Any]]:
        """Get information about running processes with optional filtering.
        
        Args:
            filter_name: Optional filter to match process names (case-insensitive substring match)
            
        Returns:
            List of dicts containing process information sorted by CPU usage.
        """
        try:
            processes = []
            
            # Get all running processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status', 'create_time', 'username']):
                try:
                    # Get process info - this can raise exceptions
                    proc_info = proc.info
                    
                    # Apply filter if provided
                    if filter_name and filter_name.lower() not in proc_info['name'].lower():
                        continue
                    
                    # Calculate memory usage in MB
                    memory_mb = 0
                    if proc_info['memory_info']:
                        memory_mb = proc_info['memory_info'].rss / (1024 * 1024)
                    
                    # Get CPU percentage (this might be 0.0 for the first call)
                    cpu_percent = proc_info['cpu_percent'] or 0.0
                    
                    # Get username safely
                    username = 'Unknown'
                    try:
                        username = proc_info['username'] or 'Unknown'
                    except (psutil.AccessDenied, KeyError):
                        username = 'Access Denied'
                    
                    # Format create time
                    create_time_str = 'Unknown'
                    if proc_info['create_time']:
                        try:
                            create_time = datetime.fromtimestamp(proc_info['create_time'])
                            create_time_str = create_time.strftime('%Y-%m-%d %H:%M:%S')
                        except (ValueError, OSError):
                            create_time_str = 'Unknown'
                    
                    process_data = {
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cpu_percent': round(cpu_percent, 1),
                        'memory_mb': round(memory_mb, 1),
                        'memory_formatted': ProcessInfoCollector._format_bytes(int(memory_mb * 1024 * 1024)),
                        'status': proc_info['status'],
                        'username': username,
                        'create_time': create_time_str
                    }
                    
                    processes.append(process_data)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Skip processes that disappeared or we can't access
                    continue
            
            # Sort processes by CPU usage (descending), then by memory usage (descending)
            processes.sort(key=lambda x: (x['cpu_percent'], x['memory_mb']), reverse=True)
            
            return processes
            
        except Exception as e:
            return [{
                'pid': 0,
                'name': 'Error',
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_formatted': '0 B',
                'status': 'Unknown',
                'username': 'Unknown',
                'create_time': 'Unknown',
                'error': str(e)
            }]
    
    @staticmethod
    def get_services() -> List[Dict[str, Any]]:
        """Get information about system services (cross-platform compatible).
        
        Returns:
            List of dicts containing service information.
        """
        try:
            services = []
            
            # This is a simplified implementation since psutil doesn't have direct service support
            # We'll look for common service processes instead
            service_patterns = [
                'sshd', 'httpd', 'nginx', 'apache', 'mysql', 'postgres', 'redis',
                'docker', 'systemd', 'init', 'cron', 'rsyslog', 'dbus', 'networkd',
                'resolved', 'bluetooth', 'cups', 'avahi', 'pulseaudio'
            ]
            
            # Get processes that match common service patterns
            for proc in psutil.process_iter(['pid', 'name', 'status', 'username']):
                try:
                    proc_info = proc.info
                    proc_name_lower = proc_info['name'].lower()
                    
                    # Check if this looks like a service process
                    is_service = False
                    for pattern in service_patterns:
                        if pattern in proc_name_lower:
                            is_service = True
                            break
                    
                    # Also include processes running as system users
                    if not is_service:
                        username = proc_info.get('username', '').lower()
                        system_users = ['root', 'system', 'daemon', 'nobody', 'www-data', 'nginx', 'apache']
                        if username in system_users:
                            is_service = True
                    
                    if is_service:
                        service_data = {
                            'name': proc_info['name'],
                            'pid': proc_info['pid'],
                            'status': proc_info['status'],
                            'username': proc_info.get('username', 'Unknown')
                        }
                        services.append(service_data)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Remove duplicates by name and sort
            seen_names = set()
            unique_services = []
            for service in services:
                if service['name'] not in seen_names:
                    seen_names.add(service['name'])
                    unique_services.append(service)
            
            unique_services.sort(key=lambda x: x['name'].lower())
            
            return unique_services
            
        except Exception as e:
            return [{
                'name': 'Error',
                'pid': 0,
                'status': 'Unknown',
                'username': 'Unknown',
                'error': str(e)
            }]
    
    @staticmethod
    def get_top_processes(limit: int = 10, sort_by: str = 'cpu') -> List[Dict[str, Any]]:
        """Get top processes sorted by CPU or memory usage.
        
        Args:
            limit: Maximum number of processes to return (default: 10)
            sort_by: Sort criteria - 'cpu' or 'memory' (default: 'cpu')
            
        Returns:
            List of dicts containing top processes information.
        """
        try:
            processes = ProcessInfoCollector.get_processes()
            
            # Sort based on criteria
            if sort_by.lower() == 'memory':
                processes.sort(key=lambda x: x['memory_mb'], reverse=True)
            else:  # Default to CPU
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            # Return limited results
            return processes[:limit]
            
        except Exception as e:
            return [{
                'pid': 0,
                'name': 'Error',
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_formatted': '0 B',
                'status': 'Unknown',
                'username': 'Unknown',
                'create_time': 'Unknown',
                'error': str(e)
            }]
    
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes into human-readable string.
        
        Args:
            bytes_value: Number of bytes to format
            
        Returns:
            Formatted string (e.g., "1.5 GB", "512 MB")
        """
        if bytes_value == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0
        size = float(bytes_value)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.1f} {units[unit_index]}"


@mcp.tool()
def get_system_info() -> str:
    """Get basic system information including OS, hostname, user, and uptime.
    
    Returns comprehensive system information in a readable format including:
    - Operating system name, version, and architecture
    - Hostname and domain information
    - Current user details
    - System uptime
    
    Returns:
        str: Formatted string containing system information
    """
    try:
        # Collect system information using SystemInfoCollector
        os_info = SystemInfoCollector.get_os_info()
        hostname_info = SystemInfoCollector.get_hostname_info()
        user_info = SystemInfoCollector.get_user_info()
        uptime = SystemInfoCollector.get_uptime()
        
        # Format the output according to design specification
        output_lines = []
        
        # Operating System information
        if 'error' not in os_info:
            os_display = f"{os_info['name']} {os_info['release']}"
            if os_info['version'] and os_info['version'] != 'Unknown':
                os_display += f" ({os_info['version']})"
            output_lines.append(f"Operating System: {os_display}")
            output_lines.append(f"Architecture: {os_info['architecture']}")
        else:
            output_lines.append("Operating System: Unable to retrieve OS information")
            output_lines.append("Architecture: Unknown")
        
        # Hostname and domain information
        if 'error' not in hostname_info:
            output_lines.append(f"Hostname: {hostname_info['hostname']}")
            output_lines.append(f"Domain: {hostname_info['domain']}")
        else:
            output_lines.append("Hostname: Unable to retrieve hostname information")
            output_lines.append("Domain: Unknown")
        
        # User information
        if 'error' not in user_info:
            output_lines.append(f"Current User: {user_info['username']}")
        else:
            output_lines.append("Current User: Unable to retrieve user information")
        
        # System uptime
        output_lines.append(f"System Uptime: {uptime}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        # Graceful degradation - return basic error message
        return f"Error retrieving system information: {str(e)}"


@mcp.tool()
def get_hardware_info() -> str:
    """Get hardware specifications including CPU, memory, and disk information.
    
    Returns comprehensive hardware information in a readable format including:
    - CPU details (model, cores, frequency)
    - Memory information (total, available, usage)
    - Disk space information for all mounted drives
    
    Returns:
        str: Formatted string containing hardware specifications
    """
    try:
        # Collect hardware information using HardwareInfoCollector
        cpu_info = HardwareInfoCollector.get_cpu_info()
        memory_info = HardwareInfoCollector.get_memory_info()
        disk_info = HardwareInfoCollector.get_disk_info()
        
        # Format the output according to design specification
        output_lines = []
        
        # CPU Information
        if 'error' not in cpu_info:
            cpu_model = cpu_info.get('model', 'Unknown')
            physical_cores = cpu_info.get('physical_cores', 0)
            logical_cores = cpu_info.get('logical_cores', 0)
            max_freq = cpu_info.get('max_frequency', 0.0)
            
            # Format CPU line
            cpu_line = f"CPU: {cpu_model}"
            if max_freq > 0:
                cpu_line += f" @ {max_freq/1000:.2f}GHz"
            output_lines.append(cpu_line)
            
            # Format cores information
            if physical_cores > 0 and logical_cores > 0:
                if physical_cores == logical_cores:
                    output_lines.append(f"CPU Cores: {physical_cores}")
                else:
                    output_lines.append(f"CPU Cores: {logical_cores} ({physical_cores} physical, {logical_cores - physical_cores} logical)")
            elif logical_cores > 0:
                output_lines.append(f"CPU Cores: {logical_cores}")
            else:
                output_lines.append("CPU Cores: Unknown")
        else:
            output_lines.append("CPU: Unable to retrieve CPU information")
            output_lines.append("CPU Cores: Unknown")
        
        # Memory Information
        if 'error' not in memory_info:
            total_mem = memory_info.get('total_formatted', '0 B')
            available_mem = memory_info.get('available_formatted', '0 B')
            output_lines.append(f"Total Memory: {total_mem}")
            output_lines.append(f"Available Memory: {available_mem}")
        else:
            output_lines.append("Total Memory: Unable to retrieve memory information")
            output_lines.append("Available Memory: Unknown")
        
        # Disk Information
        if disk_info and len(disk_info) > 0:
            # Check if first disk has error
            if 'error' in disk_info[0]:
                output_lines.append("Primary Disk: Unable to retrieve disk information")
            else:
                # Find primary disk first
                primary_disk = None
                for disk in disk_info:
                    if disk.get('is_primary', False):
                        primary_disk = disk
                        break
                
                # If no primary disk found, use the first one
                if not primary_disk and disk_info:
                    primary_disk = disk_info[0]
                
                if primary_disk:
                    device = primary_disk.get('device', 'Unknown')
                    mountpoint = primary_disk.get('mountpoint', 'Unknown')
                    filesystem = primary_disk.get('filesystem', 'Unknown')
                    total = primary_disk.get('total_formatted', '0 B')
                    free = primary_disk.get('free_formatted', '0 B')
                    
                    # Format primary disk line
                    disk_line = f"Primary Disk: {mountpoint}"
                    if filesystem and filesystem != 'Unknown':
                        disk_line += f" ({filesystem})"
                    disk_line += f" - {total} total, {free} free"
                    output_lines.append(disk_line)
                    
                    # Add additional disks if present
                    other_disks = [d for d in disk_info if not d.get('is_primary', False)]
                    if len(other_disks) > 0:
                        output_lines.append("Additional Disks:")
                        for disk in other_disks[:3]:  # Limit to first 3 additional disks
                            mountpoint = disk.get('mountpoint', 'Unknown')
                            total = disk.get('total_formatted', '0 B')
                            free = disk.get('free_formatted', '0 B')
                            output_lines.append(f"  {mountpoint}: {total} total, {free} free")
                else:
                    output_lines.append("Primary Disk: No accessible disk information")
        else:
            output_lines.append("Primary Disk: No accessible disk information")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        # Graceful degradation - return basic error message
        return f"Error retrieving hardware information: {str(e)}"


@mcp.tool()
async def get_network_info() -> str:
    """Get network configuration details including interfaces, gateway, DNS, and public IP.
    
    Returns comprehensive network information in a readable format including:
    - Active network interfaces with IP addresses and MAC addresses
    - Default gateway information
    - DNS server configuration
    - Public IP address
    
    Returns:
        str: Formatted string containing network configuration details
    """
    try:
        # Collect network information using NetworkInfoCollector
        interfaces = NetworkInfoCollector.get_interfaces()
        gateway_info = NetworkInfoCollector.get_gateway_info()
        dns_servers = NetworkInfoCollector.get_dns_info()
        
        # Handle public IP lookup separately to avoid failing the entire function
        try:
            public_ip = await NetworkInfoCollector.get_public_ip()
        except Exception as e:
            public_ip = f"Error getting public IP: {str(e)}"
        
        # Format the output according to design specification
        output_lines = []
        
        # Active Interfaces
        active_interfaces = [iface for iface in interfaces if iface.get('is_active', False)]
        if active_interfaces:
            output_lines.append("Active Interfaces:")
            for interface in active_interfaces:
                interface_name = interface.get('name', 'Unknown')
                mac_address = interface.get('mac_address', 'Unknown')
                ip_addresses = interface.get('ip_addresses', [])
                
                # Find primary IPv4 address
                primary_ipv4 = None
                for ip_info in ip_addresses:
                    if ip_info.get('family') == 'IPv4' and not ip_info.get('ip', '').startswith('127.'):
                        primary_ipv4 = ip_info
                        break
                
                if primary_ipv4:
                    ip_addr = primary_ipv4.get('ip', 'Unknown')
                    netmask = primary_ipv4.get('netmask', '')
                    
                    # Convert netmask to CIDR notation if possible
                    cidr_suffix = ""
                    if netmask:
                        try:
                            # Convert netmask to CIDR
                            cidr = sum([bin(int(x)).count('1') for x in netmask.split('.')])
                            cidr_suffix = f"/{cidr}"
                        except (ValueError, AttributeError):
                            pass
                    
                    interface_line = f"  {interface_name}: {ip_addr}{cidr_suffix}"
                    if mac_address and mac_address != 'Unknown':
                        interface_line += f" (MAC: {mac_address})"
                    output_lines.append(interface_line)
                else:
                    # No IPv4 address found, show interface name only
                    interface_line = f"  {interface_name}: No IPv4 address"
                    if mac_address and mac_address != 'Unknown':
                        interface_line += f" (MAC: {mac_address})"
                    output_lines.append(interface_line)
        else:
            output_lines.append("Active Interfaces: No active network interfaces found")
        
        # Default Gateway
        gateway_ip = gateway_info.get('gateway_ip')
        if gateway_ip:
            output_lines.append(f"Default Gateway: {gateway_ip}")
        else:
            output_lines.append("Default Gateway: Unable to determine")
        
        # DNS Servers
        if dns_servers:
            dns_list = ", ".join(dns_servers)
            output_lines.append(f"DNS Servers: {dns_list}")
        else:
            output_lines.append("DNS Servers: Unable to determine")
        
        # Public IP
        if public_ip and not public_ip.startswith("Error") and not public_ip.startswith("Unable"):
            output_lines.append(f"Public IP: {public_ip}")
        else:
            output_lines.append(f"Public IP: {public_ip}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        # Graceful degradation - return basic error message
        return f"Error retrieving network information: {str(e)}"


@mcp.tool()
def get_resource_usage() -> str:
    """Get current system resource usage including CPU, memory, disk I/O, and network I/O.
    
    Returns real-time system resource utilization information in a readable format including:
    - CPU usage percentage (overall and per-core)
    - Memory usage (virtual and swap)
    - Disk I/O statistics (read/write rates and totals)
    - Network I/O statistics (bytes sent/received per interface)
    - Load averages (on Unix-like systems)
    
    Returns:
        String containing formatted resource usage information
    """
    try:
        # Collect resource usage data
        cpu_data = ResourceUsageCollector.get_cpu_usage()
        memory_data = ResourceUsageCollector.get_memory_usage()
        disk_data = ResourceUsageCollector.get_disk_io()
        network_data = ResourceUsageCollector.get_network_io()
        
        # Format the output
        output_lines = []
        
        # CPU Usage Section
        output_lines.append("=== CPU Usage ===")
        if 'error' not in cpu_data:
            output_lines.append(f"Overall CPU: {cpu_data['overall_percent']}%")
            
            # Show per-core usage if available and reasonable number of cores
            if cpu_data['per_core_percent'] and len(cpu_data['per_core_percent']) <= 16:
                core_usage = ", ".join([f"Core {i}: {usage}%" for i, usage in enumerate(cpu_data['per_core_percent'])])
                output_lines.append(f"Per-Core: {core_usage}")
            elif cpu_data['per_core_percent']:
                output_lines.append(f"Cores: {cpu_data['core_count']} cores active")
            
            # Show frequency if available
            if cpu_data['frequency'] and cpu_data['frequency']['current'] > 0:
                freq_ghz = cpu_data['frequency']['current'] / 1000
                output_lines.append(f"CPU Frequency: {freq_ghz:.2f} GHz")
            
            # Show load average if available
            if cpu_data['load_average']:
                load_str = ", ".join([str(load) for load in cpu_data['load_average']])
                output_lines.append(f"Load Average: {load_str} (1m, 5m, 15m)")
        else:
            output_lines.append(f"CPU: Error - {cpu_data['error']}")
        
        # Memory Usage Section
        output_lines.append("\n=== Memory Usage ===")
        if 'error' not in memory_data:
            virtual = memory_data['virtual']
            output_lines.append(f"Virtual Memory: {virtual['used_formatted']} / {virtual['total_formatted']} ({virtual['percent_used']}% used)")
            output_lines.append(f"Available Memory: {virtual['available_formatted']}")
            
            # Show swap if it exists
            swap = memory_data['swap']
            if swap['total_bytes'] > 0:
                output_lines.append(f"Swap Memory: {swap['used_formatted']} / {swap['total_formatted']} ({swap['percent_used']}% used)")
            else:
                output_lines.append("Swap Memory: Not configured")
        else:
            output_lines.append(f"Memory: Error - {memory_data['error']}")
        
        # Disk I/O Section
        output_lines.append("\n=== Disk I/O ===")
        if 'error' not in disk_data:
            total = disk_data['total']
            if total and total['read_bytes'] > 0 or total['write_bytes'] > 0:
                output_lines.append(f"Total Disk I/O: Read {total['read_bytes_formatted']}, Write {total['write_bytes_formatted']}")
                output_lines.append(f"Disk Operations: {total['read_count']} reads, {total['write_count']} writes")
                
                # Show per-disk stats for active disks (limit to top 5)
                active_disks = [(name, stats) for name, stats in disk_data['per_disk'].items() 
                               if stats['read_bytes'] > 0 or stats['write_bytes'] > 0]
                if active_disks and len(active_disks) <= 5:
                    output_lines.append("Per-Disk Activity:")
                    for disk_name, stats in active_disks[:5]:
                        output_lines.append(f"  {disk_name}: Read {stats['read_bytes_formatted']}, Write {stats['write_bytes_formatted']}")
            else:
                output_lines.append("Disk I/O: No recent activity")
        else:
            output_lines.append(f"Disk I/O: Error - {disk_data['error']}")
        
        # Network I/O Section
        output_lines.append("\n=== Network I/O ===")
        if 'error' not in network_data:
            total = network_data['total']
            if total and (total['bytes_sent'] > 0 or total['bytes_recv'] > 0):
                output_lines.append(f"Total Network I/O: Sent {total['bytes_sent_formatted']}, Received {total['bytes_recv_formatted']}")
                output_lines.append(f"Network Packets: {total['packets_sent']} sent, {total['packets_recv']} received")
                
                # Show errors if any
                if total['errin'] > 0 or total['errout'] > 0 or total['dropin'] > 0 or total['dropout'] > 0:
                    output_lines.append(f"Network Errors: {total['errin']} in, {total['errout']} out, {total['dropin']} dropped in, {total['dropout']} dropped out")
                
                # Show per-interface stats for active interfaces (limit to top 5)
                active_interfaces = [(name, stats) for name, stats in network_data['per_interface'].items() 
                                   if stats['bytes_sent'] > 0 or stats['bytes_recv'] > 0]
                if active_interfaces and len(active_interfaces) <= 5:
                    output_lines.append("Per-Interface Activity:")
                    for interface_name, stats in active_interfaces[:5]:
                        output_lines.append(f"  {interface_name}: Sent {stats['bytes_sent_formatted']}, Received {stats['bytes_recv_formatted']}")
            else:
                output_lines.append("Network I/O: No recent activity")
        else:
            output_lines.append(f"Network I/O: Error - {network_data['error']}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        # Graceful degradation - return basic error message
        return f"Error retrieving resource usage information: {str(e)}"


@mcp.tool()
def get_process_info(filter: str = "") -> str:
    """Get information about running processes with optional filtering.
    
    Returns information about currently running processes in a readable table format including:
    - Process ID (PID)
    - Process name
    - CPU usage percentage
    - Memory usage (MB and formatted)
    - Process status
    - Username (owner)
    - Process creation time
    
    Args:
        filter: Optional filter to match process names (case-insensitive substring match).
                For example: "chrome", "python", "system" to show only matching processes.
    
    Returns:
        String containing formatted table of process information sorted by CPU usage
    """
    try:
        # Validate and sanitize the filter parameter for security
        if filter is not None and not isinstance(filter, str):
            filter = str(filter)
        
        # Limit filter length to prevent potential issues
        if filter and len(filter) > 100:
            filter = filter[:100]
        
        # Remove potentially dangerous characters (basic sanitization)
        if filter:
            # Allow only alphanumeric, spaces, hyphens, underscores, and dots
            import re
            filter = re.sub(r'[^a-zA-Z0-9\s\-_\.]', '', filter).strip()
        
        # Collect process information using ProcessInfoCollector
        processes = ProcessInfoCollector.get_processes(filter_name=filter)
        
        # Handle error case
        if not processes:
            if filter:
                return f"No processes found matching filter: '{filter}'"
            else:
                return "No processes found or unable to access process information"
        
        # Check if we got an error result
        if len(processes) == 1 and 'error' in processes[0]:
            return f"Error retrieving process information: {processes[0]['error']}"
        
        # Store original count before truncation for summary
        original_process_count = len(processes)
        
        # Format the output as a readable table
        output_lines = []
        
        # Add header with filter info if applicable
        if filter:
            output_lines.append(f"Running Processes (filtered by '{filter}'):")
        else:
            # Limit to top 20 processes to keep output manageable
            if len(processes) > 20:
                processes = processes[:20]
                output_lines.append("Running Processes (Top 20 by CPU usage):")
            else:
                output_lines.append("Running Processes:")
        
        output_lines.append("")
        
        # Table header
        header = f"{'PID':<8} {'Name':<25} {'CPU%':<6} {'Memory':<12} {'Status':<12} {'User':<15} {'Started':<19}"
        output_lines.append(header)
        output_lines.append("-" * len(header))
        
        # Process rows
        for proc in processes:
            # Truncate long process names for better table formatting
            name = proc['name'][:24] if len(proc['name']) > 24 else proc['name']
            
            # Truncate long usernames
            username = proc['username'][:14] if len(proc['username']) > 14 else proc['username']
            
            # Format the row
            row = (f"{proc['pid']:<8} "
                   f"{name:<25} "
                   f"{proc['cpu_percent']:<6} "
                   f"{proc['memory_formatted']:<12} "
                   f"{proc['status']:<12} "
                   f"{username:<15} "
                   f"{proc['create_time']:<19}")
            
            output_lines.append(row)
        
        # Add summary information
        output_lines.append("")
        if filter:
            output_lines.append(f"Total matching processes: {len(processes)}")
        else:
            # Use the original count before truncation
            if original_process_count > 20:
                output_lines.append(f"Showing top 20 of {original_process_count} total processes")
            else:
                output_lines.append(f"Total processes: {original_process_count}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        # Graceful degradation - return basic error message
        return f"Error retrieving process information: {str(e)}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')