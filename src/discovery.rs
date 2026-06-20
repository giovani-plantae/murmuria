//! Service discovery (mDNS / Bonjour) so clients can find this server on the LAN
//! without a hardcoded address.
//!
//! Two kinds of client are served by a single advertisement:
//! - Native clients can browse the DNS-SD service type [`SERVICE_TYPE`].
//! - Browsers can't browse DNS-SD, but they *do* resolve `.local` names through
//!   the OS mDNS responder — so we advertise a fixed hostname [`HOSTNAME`], and a
//!   browser extension simply fetches `http://murmuria.local:<port>`.

use std::net::{IpAddr, UdpSocket};

use mdns_sd::{ServiceDaemon, ServiceInfo};

/// DNS-SD service type advertised on the local network (for native clients).
pub const SERVICE_TYPE: &str = "_murmuria._tcp.local.";

/// Fixed hostname advertised so browser clients can resolve `murmuria.local`
/// regardless of the machine's own hostname or IP.
pub const HOSTNAME: &str = "murmuria.local.";

/// Advertises this server on the LAN. The returned [`ServiceDaemon`] keeps the
/// mDNS responder alive; drop it to stop advertising. Returns `None` (and logs)
/// if mDNS can't be started, so discovery is best-effort and never blocks boot.
pub fn advertise(port: u16) -> Option<ServiceDaemon> {
    let daemon = match ServiceDaemon::new() {
        Ok(daemon) => daemon,
        Err(error) => {
            eprintln!("murmuria: mDNS disabled ({error}); clients must use a configured address");
            return None;
        }
    };

    // On a multi-homed host (Docker bridges, VPNs) enable_addr_auto announces
    // every interface — including docker0 (172.17.x), which LAN clients can't
    // reach. Announce a single routable LAN address instead, falling back to
    // every interface only when one can't be determined.
    let no_properties: &[(&str, &str)] = &[];
    let advertise_ip = resolve_advertise_ip();
    let info = match advertise_ip {
        Some(ip) => ServiceInfo::new(SERVICE_TYPE, "murmuria", HOSTNAME, ip, port, no_properties),
        None => ServiceInfo::new(SERVICE_TYPE, "murmuria", HOSTNAME, "", port, no_properties)
            .map(|info| info.enable_addr_auto()),
    };
    let service = match info {
        Ok(service) => service,
        Err(error) => {
            eprintln!("murmuria: mDNS service build failed ({error})");
            return None;
        }
    };

    match daemon.register(service) {
        Ok(()) => {
            let address = advertise_ip
                .map(|ip| ip.to_string())
                .unwrap_or_else(|| "all interfaces".to_string());
            println!(
                "murmuria: advertising mDNS as http://{}:{port} (addr {address})",
                HOSTNAME.trim_end_matches('.')
            );
            Some(daemon)
        }
        Err(error) => {
            eprintln!("murmuria: mDNS registration failed ({error})");
            None
        }
    }
}

/// The address to advertise over mDNS: `MURMURIA_ADVERTISE_IP` when set, otherwise
/// the primary LAN IP. `None` falls back to announcing every interface.
fn resolve_advertise_ip() -> Option<IpAddr> {
    if let Ok(value) = std::env::var("MURMURIA_ADVERTISE_IP") {
        match value.trim().parse() {
            Ok(ip) => return Some(ip),
            Err(_) => eprintln!("murmuria: ignoring invalid MURMURIA_ADVERTISE_IP={value:?}"),
        }
    }
    primary_lan_ip()
}

/// Finds the primary LAN IP by asking the OS which local address it would use to
/// reach a public host — no packets are sent, it only resolves the route. This
/// skips Docker/VPN interfaces, which don't carry the default route.
fn primary_lan_ip() -> Option<IpAddr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    socket.local_addr().ok().map(|addr| addr.ip())
}
