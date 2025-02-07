# Salesforce Metadata Types

This page provides an overview of various **Salesforce Metadata Types** used for configuration, automation, and customization in Salesforce development.

## **Introduction**
Salesforce metadata types define the different kinds of metadata components that can be deployed, retrieved, or modified using tools like **Metadata API**, **SFDX**, and **Change Sets**. This README provides an overview of key metadata types and their use cases.

---

## **Metadata Types Overview**

### **1. AI & Automation**
- **AIApplication** – Represents an AI-powered application in Salesforce.
- **AIApplicationConfig** – Stores configuration settings for AI applications.
- **ApexClass** – Represents a class written in **Apex**, Salesforce's programming language.
- **ApexComponent** – Represents a **Visualforce component** used in UI development.
- **ApexTrigger** – An **Apex script** that executes before or after record operations (Insert, Update, Delete).

### **2. User Interface & Experience**
- **AuraDefinitionBundle** – Contains **Aura components**, used for building Lightning components.
- **LightningComponentBundle** – Represents **Lightning Web Components (LWC)** for modern UI development.
- **FlexiPage** – Defines **Lightning App Pages**, **Record Pages**, and **Home Pages** in the Lightning Experience.
- **HomePageLayout** – Defines layouts for the **Salesforce Home Page**.
- **AppMenu** – Configures the **App Launcher Menu**.

### **3. Workflow & Automation**
- **Flow** – Represents **Flow Builder Flows** for automating business processes.
- **ApprovalProcess** – Defines **approval workflows** for records.
- **AssignmentRules** – Sets up **record assignment rules**, e.g., for Leads or Cases.
- **AutoResponseRules** – Defines **automatic email responses** for incoming leads or cases.
- **EscalationRules** – Defines escalation conditions for **cases**.

### **4. Security & Permissions**
- **AuthProvider** – Defines **authentication providers** (e.g., Google, Facebook, or SAML providers).
- **PermissionSet** – Defines **custom user permissions** beyond profile-level permissions.
- **MutingPermissionSet** – Muting of permissions for specific users or profiles.

### **5. Data & Object Configuration**
- **CustomObject** – Represents a **custom Salesforce object**.
- **CustomMetadata** – Stores custom configuration settings (like Custom Objects but for metadata).
- **GlobalValueSet** – Defines a **picklist value set** shared across multiple objects.
- **MatchingRules** – Defines **rules for duplicate record detection**.
- **DuplicateRule** – Prevents duplicate records based on **Matching Rules**.

### **6. Email & Communication**
- **ApexEmailNotifications** – Defines **custom email notifications** via Apex.
- **EmailTemplate** – Represents **email templates** for automation.
- **Letterhead** – Defines **email branding** (logos, colors, etc.).
- **EmailServicesFunction** – Represents an **Apex Email Service** that processes incoming emails.

### **7. Reporting & Dashboards**
- **Dashboard** – Represents **Salesforce Dashboards**.
- **Report** – Represents a **custom Salesforce Report**.
- **AnalyticSnapshot** – Stores **historical data snapshots** for reports.

### **8. Digital Engagement**
- **LiveChatSensitiveDataRule** – Manages **sensitive data redaction** in live chat.
- **MessagingChannel** – Defines **messaging services** (SMS, WhatsApp, Facebook Messenger).
- **ExperienceContainer** – Manages **Experience Cloud (Community) components**.

### **9. Integrations & External Data**
- **ConnectedApp** – Defines **OAuth-connected applications**.
- **ExternalDataSource** – Represents **external databases** that Salesforce can access.
- **NamedCredential** – Stores **external system authentication credentials** securely.
- **ExternalServiceRegistration** – Registers **external services** for API calls.

---

## **Wildcard (`*`) in `<members>`**
The `*` wildcard inside `<members>*</members>` means **all components** of that metadata type are included. For example:

```xml
<types>
    <members>*</members>
    <name>ApexClass</name>
</types>
```
This means **all Apex classes** will be included in the deployment or retrieval.

---

## **Conclusion**
These metadata types represent various Salesforce configurations, customizations, and automations. They can be retrieved or deployed using **Salesforce Metadata API** or **SFDX**.

For more information, refer to the **[Salesforce Metadata API Documentation](https://developer.salesforce.com/docs/atlas.en-us.api_meta.meta/api_meta/meta_intro.htm)**.

---
