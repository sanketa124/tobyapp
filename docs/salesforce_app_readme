# Salesforce Login and Open a Specific Lightning App

This guide explains how to authenticate into Salesforce and navigate to a specific Lightning App.

## Prerequisites

- A Salesforce account with API access
- A connected app in Salesforce with OAuth enabled
- Consumer Key and Consumer Secret from the connected app
- Salesforce instance URL
- Postman or a CLI tool like `cURL` (for testing authentication)

## Steps

### 1. Create a Connected App
1. Log in to Salesforce.
2. Navigate to **Setup** > **App Manager**.
3. Click **New Connected App**.
4. Fill in required details:
   - **Enable OAuth Settings**
   - Add callback URL: `https://login.salesforce.com/services/oauth2/callback`
   - Select OAuth Scopes like `Full Access` or `Access and Manage Your Data`
5. Save and note down the **Consumer Key** and **Consumer Secret**.

### 2. Get an OAuth Access Token

Use `cURL` or Postman to get an access token:

```sh
curl -X POST https://login.salesforce.com/services/oauth2/token \
     -d "grant_type=password" \
     -d "client_id=YOUR_CONSUMER_KEY" \
     -d "client_secret=YOUR_CONSUMER_SECRET" \
     -d "username=YOUR_SALESFORCE_USERNAME" \
     -d "password=YOUR_SALESFORCE_PASSWORD"
```

This will return an access token:
```json
{
  "access_token": "YOUR_ACCESS_TOKEN",
  "instance_url": "https://yourInstance.salesforce.com",
  "token_type": "Bearer"
}
```

### 3. Open a Specific Lightning App

Use the following URL format to open a Lightning App:

```
https://yourInstance.lightning.force.com/lightning/app/YOUR_APP_ID
```

To find the `YOUR_APP_ID`:
1. Open Salesforce.
2. Navigate to **App Manager**.
3. Find your Lightning App and note the URL's last part (after `/lightning/app/`).

### 4. Automate Login & Navigation

If you need to automate the login and redirection, you can:
- Embed the Salesforce URL in an iframe (if permitted)
- Use Selenium or Puppeteer for automation
- Build a custom login script using OAuth

## Troubleshooting

- **Invalid credentials**: Check username, password, and security token (if required).
- **App not found**: Ensure the App ID is correct.
- **Insufficient permissions**: Verify OAuth scopes and user permissions.

## References
- [Salesforce OAuth Documentation](https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/quickstart_oauth.htm)
- [Salesforce Lightning App Documentation](https://developer.salesforce.com/docs/component-library/overview/components)

---
This README provides a step-by-step guide for logging into Salesforce and opening a specific Lightning App programmatically.
