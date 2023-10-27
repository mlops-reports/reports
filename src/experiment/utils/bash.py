import socket

def is_server_running(host, port):
    '''The function `is_server_running` checks if a server is running on a specified host and port by
    attempting to establish a TCP connection.
    
    Parameters
    ----------
    host
        The `host` parameter is the IP address or hostname of the server you want to check if it is
    running.
    port
        The `port` parameter is the port number on which the server is expected to be running. It is an
    integer value that represents the port number.
    
    Returns
    -------
        a boolean value. It returns True if the server is running and can be connected to at the specified
    host and port, and False if the server is not running or cannot be connected to.
    
    '''
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)  # Set a timeout for the connection attempt
            s.connect((host, port))
        return True
    except (socket.timeout, ConnectionRefusedError):
        return False