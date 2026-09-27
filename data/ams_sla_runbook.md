# SAP AMS Service Level Agreement & Operational Runbook

Customer: Global Manufacturing Operations | Service Provider: NTT DATA AMS | Version: 2.4 (Effective:
2026-2027)


## 1. Service Scope and Operational Objectives

This document establishes the binding Service Level Agreements (SLAs), classification rubrics, and emergency
operational procedures governing the SAP Application Management Services (AMS) contract. The covered scope
includes production instances of SAP S/4HANA (Client 100/200), covering functional modules FI (Financials), SD
(Sales & Distribution), MM (Materials Management), as well as technical domains ABAP development and Basis
system administration. The primary objective is to guarantee business continuity, rapid containment of operational
stoppages, and strict adherence to ITIL-aligned restoration procedures.


## 2. Incident Priority & Response SLA Matrix

All incoming support tickets must be triaged according to the severity matrix below. Target resolution times represent
maximum elapsed time from ticket creation to technical restoration.


<table>
<tr>
<th>Priority</th>
<th>Severity Classification</th>
<th>Target Response</th>
<th>Target Resolution</th>
<th>Escalation Authority</th>
</tr>
<tr>
<td>P1</td>
<td>Critical Outage: Core plant or site completely halted</td>
<td>15 minutes</td>
<td>2 hours</td>
<td>L3 On-Call Lead + Service Delivery Manager</td>
</tr>
<tr>
<td>P2</td>
<td>Major Degradation: Entire department workflow blocked</td>
<td>30 minutes</td>
<td>4 hours</td>
<td>L2 Specialist + Module Duty Manager</td>
</tr>
<tr>
<td>P3</td>
<td>Medium Issue: Workaround exists or non-critical task delayed</td>
<td>2 hours</td>
<td>24 hours</td>
<td>Functional Lead (FI/SD/MM/Basis)</td>
</tr>
<tr>
<td>P4</td>
<td>Minor / Routine: Single user inquiry or cosmetic defect</td>
<td>4 hours</td>
<td>72 hours</td>
<td>Service Desk L1 Team</td>
</tr>
</table>


## 3. Emergency Transport Procedures (STMS)

Emergency software corrections and hotfixes requiring transport release outside standard weekly maintenance
windows must follow strict governance. An Emergency Change Request (CHG) must be approved in writing by the
Basis Lead and the functional Module Owner. All emergency transports must be verified in QAS quality assurance
prior to importing into PRD. Bypassing the standard transport route (DEV -> QAS -> PRD) is strictly prohibited under
any circumstances.


## 4. Emergency Contact Directory

24/7 Global AMS Bridge: +1 (800) 555-0199 | Major Incident Commander: incident-command@example.com
Service Delivery Escalations: ams-delivery-lead@example.com
