//! Service discovery (mDNS / Bonjour) so clients can find this server on the LAN
//! without a hardcoded address.
//!
//! Two kinds of client are served by a single advertisement:
//! - Native clients can browse the DNS-SD service type [`SERVICE_TYPE`].
//! - Browsers can't browse DNS-SD, but they *do* resolve `.local` names through
//!   the OS mDNS responder — so we advertise a fixed hostname [`HOSTNAME`], and a
//!   browser extension simply fetches `http://murmuria.local:<port>`.

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

    // Empty address + enable_addr_auto: the daemon detects the host's interface
    // IPs and keeps the announced A records in sync as the network changes.
    let no_properties: &[(&str, &str)] = &[];
    let service =
        match ServiceInfo::new(SERVICE_TYPE, "murmuria", HOSTNAME, "", port, no_properties) {
            Ok(service) => service.enable_addr_auto(),
            Err(error) => {
                eprintln!("murmuria: mDNS service build failed ({error})");
                return None;
            }
        };

    match daemon.register(service) {
        Ok(()) => {
            println!(
                "murmuria: advertising mDNS as http://{}:{port}",
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
