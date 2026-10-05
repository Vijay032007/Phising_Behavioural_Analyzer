# Phase 2 — Live Behavioral Analyzer

This folder intentionally does not automatically visit arbitrary URLs yet.

The next implementation will use an isolated Playwright browser with:
- no persistent profile
- no saved credentials/cookies
- navigation timeout
- no downloads
- no file uploads
- controlled interactions only
- network request logging

Feature groups:
1. URL
2. HTML/DOM
3. forms and form destinations
4. JavaScript and obfuscation indicators
5. redirect chain
6. iframes
7. external scripts/resources
8. network/domain graph
9. brand similarity
10. screenshot/DOM consistency

The behavioral extractor must produce exactly the feature schema expected by the final hybrid model. It should not be allowed to silently invent values when a feature cannot be collected.
